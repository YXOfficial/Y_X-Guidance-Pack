import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from modules import devices, shared

try:
    import modules.sd_hijack_clip as sd_hijack_clip
except Exception:
    sd_hijack_clip = None

try:
    import modules.sd_emphasis as sd_emphasis
except Exception:
    sd_emphasis = None


@dataclass
class TPSEmbedding:
    cross_attn: torch.Tensor
    pooled: Optional[torch.Tensor] = None


@dataclass
class TPSOConfig:
    enabled: bool = False
    num_variants: int = 4
    kappa: float = 0.80
    sigma: float = 0.01
    lambda_div: float = 5.0
    r_ratio: float = 0.4
    num_opt_steps: int = 30
    lr: float = 1e-2


def _get_transformer(embedder):
    wrapped = getattr(embedder, "wrapped", None)
    if wrapped is None:
        return None
    return getattr(wrapped, "transformer", None)


def _get_embedding_layer(embedder):
    wrapped = getattr(embedder, "wrapped", None)
    if wrapped is None:
        return None

    transformer = getattr(wrapped, "transformer", None)
    if transformer is not None:
        if hasattr(transformer, "get_input_embeddings"):
            return transformer.get_input_embeddings()
        text_model = getattr(transformer, "text_model", None)
        if text_model is not None and hasattr(text_model, "get_input_embeddings"):
            return text_model.get_input_embeddings()

    token_embedding = getattr(getattr(wrapped, "model", None), "token_embedding", None)
    if token_embedding is not None:
        return getattr(token_embedding, "wrapped", token_embedding)

    return None


def _resolve_clip_embedder(clip_wrapper):
    if clip_wrapper is None:
        return None

    candidates: List = []

    wrapped = getattr(clip_wrapper, "wrapped", None)
    if hasattr(clip_wrapper, "encode_with_transformers") and wrapped is not None:
        candidates.append(clip_wrapper)

    embedders = getattr(clip_wrapper, "embedders", None)
    if embedders is not None:
        if isinstance(embedders, Iterable):
            candidates.extend(list(embedders))

    for embedder in candidates:
        transformer = _get_transformer(embedder)
        embedding_layer = _get_embedding_layer(embedder)
        if hasattr(embedder, "encode_with_transformers") and transformer is not None and embedding_layer is not None:
            return embedder

    logging.error(
        "TPSO: unable to resolve a compatible CLIP embedder from %s", type(clip_wrapper).__name__
    )
    return None


def _encode_chunk_with_embeds(
    clip_wrapper,
    embedder,
    tokens_tensor: torch.Tensor,
    token_lists: Sequence[Sequence[int]],
    multipliers: Sequence[Sequence[float]],
    embedding_layer,
    offsets: Optional[torch.Tensor] = None,
):
    # Adapted from FrozenCLIPEmbedderWithCustomWords.process_tokens but allows offsets via embedding hooks.
    if embedder.id_end != embedder.id_pad:
        for batch_pos in range(tokens_tensor.shape[0]):
            token_row = tokens_tensor[batch_pos]
            end_positions = (token_row == embedder.id_end).nonzero(as_tuple=False)
            if end_positions.numel() > 0:
                index = end_positions[0, 0].item()
                token_row[index + 1 :] = embedder.id_pad

    hook_handle = None
    if offsets is not None:
        if offsets.shape != (tokens_tensor.shape[0], tokens_tensor.shape[1], offsets.shape[-1]):
            logging.debug(
                "TPSO: offset shape mismatch, expected %s got %s", tokens_tensor.shape, offsets.shape
            )
        else:
            def _offset_hook(module, args, output):
                try:
                    return output + offsets
                except Exception:
                    logging.exception("TPSO: embedding offset hook failed")
                    return output

            hook_handle = embedding_layer.register_forward_hook(_offset_hook)

    try:
        z = embedder.encode_with_transformers(tokens_tensor)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    offsets_aligned = None
    if offsets is not None:
        if offsets.shape[:2] != z.shape[:2]:
            logging.error(
                "TPSO: offset/text shape mismatch after encoding; offsets %s, encoded %s",
                offsets.shape,
                z.shape,
            )
            raise RuntimeError("TPSO offsets shape mismatch after encoding")

        offsets_aligned = offsets
        if offsets.shape != z.shape:
            offsets_aligned = offsets.expand_as(z)

    pooled = getattr(z, "pooled", None)

    emphasis = None
    emphasis_module = sd_emphasis if sd_emphasis is not None else sd_hijack_clip
    if emphasis_module is not None:
        get_emphasis = getattr(emphasis_module, "get_current_option", None)
        if callable(get_emphasis):
            try:
                emphasis = get_emphasis(shared.opts.emphasis)()
            except Exception:
                logging.exception("TPSO: emphasis option resolution failed; continuing without emphasis")
        elif emphasis_module is not None:
            logging.debug(
                "TPSO: emphasis helper not available on %s; skipping emphasis",
                getattr(emphasis_module, "__name__", type(emphasis_module).__name__),
            )

    if emphasis is not None:
        emphasis.tokens = token_lists
        emphasis.multipliers = torch.asarray(multipliers).to(devices.device)
        emphasis.z = z
        emphasis.after_transformers()
        z = emphasis.z
        if pooled is not None:
            z.pooled = pooled

    if offsets_aligned is not None:
        z = z + offsets_aligned.to(device=z.device, dtype=z.dtype)
        if pooled is not None and hasattr(z, "pooled"):
            z.pooled = pooled

    return z


def _embedding_representation(embedding: TPSEmbedding) -> torch.Tensor:
    if embedding.pooled is not None:
        return embedding.pooled
    return embedding.cross_attn.mean(dim=1)


def _build_embeddings_from_chunks(chunks: List[torch.Tensor]) -> torch.Tensor:
    return torch.hstack(chunks) if len(chunks) > 1 else chunks[0]


def _collect_tpso_embeddings(
    clip_wrapper,
    embedder,
    batch_chunks,
    embedding_layer,
    eps_params: Optional[List[List[torch.nn.Parameter]]],
):
    chunk_outputs = []
    for idx, chunk in enumerate(batch_chunks):
        tokens = torch.asarray([x.tokens for x in chunk]).to(devices.device)
        multipliers = [x.multipliers for x in chunk]
        token_lists = [x.tokens for x in chunk]

        offsets = None
        if eps_params is not None:
            offsets = eps_params[idx]

        encoded = _encode_chunk_with_embeds(
            clip_wrapper,
            embedder,
            tokens,
            token_lists,
            multipliers,
            embedding_layer,
            offsets=offsets,
        )
        chunk_outputs.append(encoded)

    pooled = getattr(chunk_outputs[0], "pooled", None) if chunk_outputs else None
    cross = _build_embeddings_from_chunks(chunk_outputs)
    return TPSEmbedding(cross_attn=cross, pooled=pooled)


def optimize_tpso_offsets(clip_wrapper, prompts: List[str], config: TPSOConfig) -> Tuple[TPSEmbedding, List[TPSEmbedding]]:
    if not config.enabled:
        raise RuntimeError("TPSO is disabled in configuration.")

    if sd_hijack_clip is None:
        raise RuntimeError("TPSO dependencies are unavailable")

    embedder = _resolve_clip_embedder(clip_wrapper)
    if embedder is None or not hasattr(embedder, "wrapped"):
        logging.error("TPSO: unable to resolve a compatible CLIP embedder from %s", type(clip_wrapper).__name__)
        raise RuntimeError("TPSO embedder resolution failed")

    batch_chunks, _ = clip_wrapper.process_texts(prompts)
    chunk_count = max(len(x) for x in batch_chunks)

    embedding_layer = _get_embedding_layer(embedder)
    if embedding_layer is None:
        logging.error("TPSO: no embedding layer available on %s", type(embedder).__name__)
        raise RuntimeError("TPSO embedder resolution failed")
    embedding_layer = embedding_layer.to(devices.device)
    attention_masks = []
    prepared_chunks = []
    for i in range(chunk_count):
        batch_chunk = [chunks[i] if i < len(chunks) else clip_wrapper.empty_chunk() for chunks in batch_chunks]
        tokens = torch.asarray([x.tokens for x in batch_chunk]).to(devices.device)
        mask = (tokens != embedder.id_pad).long()
        attention_masks.append(mask)
        prepared_chunks.append(batch_chunk)

    with torch.no_grad():
        pivot_embedding = _collect_tpso_embeddings(
            clip_wrapper,
            embedder,
            prepared_chunks,
            embedding_layer,
            eps_params=None,
        )

    embed_dim = getattr(embedding_layer, "embedding_dim", None)
    if embed_dim is None:
        weight = getattr(embedding_layer, "weight", None)
        if weight is None:
            logging.error("TPSO: embedding layer lacks weight and embedding_dim attributes")
            raise RuntimeError("TPSO embedder resolution failed")
        embed_dim = weight.shape[1]
    weight_dtype = getattr(getattr(embedding_layer, "weight", None), "dtype", torch.float32)

    eps_params: List[List[torch.nn.Parameter]] = []
    for _ in range(config.num_variants):
        variant_params = []
        for attn_mask in attention_masks:
            eps = torch.nn.Parameter(
                torch.randn(
                    attn_mask.shape[0],
                    attn_mask.shape[1],
                    embed_dim,
                    device=devices.device,
                    dtype=weight_dtype,
                )
                * 1e-4
            )
            eps.requires_grad_(True)
            variant_params.append(eps)
        eps_params.append(variant_params)

    optimizer = torch.optim.Adam([p for params in eps_params for p in params], lr=config.lr)

    for step in range(config.num_opt_steps):
        optimizer.zero_grad()
        with torch.enable_grad():
            variant_embeddings: List[TPSEmbedding] = []
            for params in eps_params:
                variant_embeddings.append(
                    _collect_tpso_embeddings(
                        clip_wrapper,
                        embedder,
                        prepared_chunks,
                        embedding_layer,
                        eps_params=params,
                    )
                )

            pivot_repr = _embedding_representation(pivot_embedding)
            variant_reprs = [_embedding_representation(v) for v in variant_embeddings]

            semantic_loss = 0.0
            for repr_k in variant_reprs:
                cos_vals = F.cosine_similarity(pivot_repr, repr_k, dim=1)
                mean_cos = cos_vals.mean()
                semantic_loss = semantic_loss + torch.clamp(torch.abs(mean_cos - config.kappa) - config.sigma, min=0.0)

            diversity_loss = 0.0
            n = len(variant_reprs)
            if n > 1:
                count = 0
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        cos_vals = F.cosine_similarity(variant_reprs[i], variant_reprs[j], dim=1)
                        diversity_loss = diversity_loss + cos_vals.mean()
                        count += 1
                diversity_loss = diversity_loss / max(count, 1)

            loss = semantic_loss + config.lambda_div * diversity_loss
            if not loss.requires_grad:
                logging.error("TPSO: loss is not connected to offsets; aborting optimization")
                raise RuntimeError("TPSO loss graph is detached from offsets")
            loss.backward()
        optimizer.step()

        logging.debug(
            "TPSO step %s: semantic=%.4f, diversity=%.4f, total=%.4f",
            step + 1,
            float(semantic_loss.detach().cpu()),
            float(diversity_loss.detach().cpu()),
            float(loss.detach().cpu()),
        )

    final_variants: List[TPSEmbedding] = []
    with torch.no_grad():
        for params in eps_params:
            final_variants.append(
                _collect_tpso_embeddings(
                    clip_wrapper,
                    embedder,
                    prepared_chunks,
                    embedding_layer,
                    eps_params=params,
                )
            )

    _log_tpso_stats(pivot_embedding, final_variants)
    return pivot_embedding, final_variants


def _log_tpso_stats(pivot: TPSEmbedding, variants: List[TPSEmbedding]):
    try:
        pivot_repr = _embedding_representation(pivot)
        variant_reprs = [_embedding_representation(v) for v in variants]

        sims = [
            float(F.cosine_similarity(pivot_repr, vk, dim=1).mean().detach().cpu()) for vk in variant_reprs
        ]
        logging.info("TPSO semantic similarities vs pivot: %s", sims)

        pairwise = []
        for i in range(len(variant_reprs)):
            for j in range(i + 1, len(variant_reprs)):
                cos_vals = F.cosine_similarity(variant_reprs[i], variant_reprs[j], dim=1)
                pairwise.append(float(cos_vals.mean().detach().cpu()))
        logging.info("TPSO pairwise variant cosine similarities: %s", pairwise)
    except Exception:
        logging.exception("TPSO stats logging failed")


def compute_alpha(step_index: int, total_steps: int, r_ratio: float) -> float:
    if total_steps <= 0:
        return 0.0
    clamp_r = max(1e-6, min(1.0, r_ratio))
    t0 = int((1.0 - clamp_r) * total_steps)
    if step_index < t0:
        return 0.0
    alpha = (step_index - t0) / (clamp_r * total_steps)
    return float(max(0.0, min(1.0, alpha)))


def blend_embeddings(pivot: TPSEmbedding, variant: TPSEmbedding, alpha: float) -> TPSEmbedding:
    alpha_tensor = torch.tensor(alpha, device=pivot.cross_attn.device, dtype=pivot.cross_attn.dtype)
    cross = pivot.cross_attn.lerp(variant.cross_attn, alpha_tensor)
    pooled = None
    if pivot.pooled is not None and variant.pooled is not None:
        pooled = pivot.pooled.lerp(variant.pooled, alpha_tensor)
    return TPSEmbedding(cross_attn=cross, pooled=pooled)



def make_tpso_conditioning_modifier(
    pivot: TPSEmbedding,
    variants: List[TPSEmbedding],
    config: TPSOConfig,
    total_steps: int,
):
    try:
        from ldm_patched.modules.conds import CONDCrossAttn, CONDRegular
    except Exception:
        CONDCrossAttn = None
        CONDRegular = None

    step_state = {"i": 0}
    variant_choice = variants[0] if variants else pivot

    def modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        alpha = compute_alpha(step_state["i"], total_steps, config.r_ratio)
        step_state["i"] += 1
        if alpha <= 0.0 or CONDCrossAttn is None or cond is None:
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        blended = blend_embeddings(pivot, variant_choice, alpha)

        for entry in cond:
            entry_cross = entry.get("cross_attn")
            if entry_cross is not None:
                entry_cross = blended.cross_attn.to(device=entry_cross.device, dtype=entry_cross.dtype)
                entry["cross_attn"] = entry_cross
                if "model_conds" in entry and "c_crossattn" in entry["model_conds"]:
                    entry["model_conds"]["c_crossattn"] = CONDCrossAttn(entry_cross)
            if blended.pooled is not None and "model_conds" in entry and entry_cross is not None:
                pooled_target = entry["model_conds"].get("y") or entry["model_conds"].get("c_crossattn")
                device = getattr(pooled_target, "cond", entry_cross).device if pooled_target is not None else entry_cross.device
                pooled_tensor = blended.pooled.to(device=device)
                entry["model_conds"]["y"] = CONDRegular(pooled_tensor)
                entry["pooled_output"] = pooled_tensor
        return model, x, timestep, uncond, cond, cond_scale, model_options, seed

    return modifier

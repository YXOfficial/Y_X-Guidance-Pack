import logging
import math
import random
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import build_laplacian_pyramid


class GuidanceState:
    def __init__(self, base_prediction: torch.Tensor, guidance_term: torch.Tensor):
        self.base_prediction = base_prediction
        self.guidance_term = guidance_term

    @property
    def prediction(self) -> torch.Tensor:
        return self.base_prediction + self.guidance_term


class GuidancePipeline:
    def __init__(self):
        self.base_builder: Callable = default_base_builder
        self.modifiers: Dict[str, Callable] = {}

    def set_base_builder(self, builder: Callable):
        self.base_builder = builder

    def add_modifier(self, name: str, modifier: Callable):
        self.modifiers[name] = modifier

    def run(self, args):
        state = self.base_builder(args)
        for modifier in self.modifiers.values():
            state = modifier(args, state)
        return state.prediction


def default_base_builder(args) -> GuidanceState:
    uncond_denoised = args["uncond_denoised"]
    cond_denoised = args["cond_denoised"]
    cond_scale = args["cond_scale"]
    guidance_term = (cond_denoised - uncond_denoised) * cond_scale
    return GuidanceState(uncond_denoised, guidance_term)


def ensure_guidance_pipeline(model) -> GuidancePipeline:
    if not hasattr(model, "_guidance_pipeline"):
        pipeline = GuidancePipeline()
        model._guidance_pipeline = pipeline
        model.set_model_sampler_post_cfg_function(pipeline.run, "custom_guidance_pipeline")
    return model._guidance_pipeline


def get_initial_sigma(model) -> float:
    try:
        return model.model.model_sampling.sigma_max
    except AttributeError:
        logging.warning("Custom Guidance: Could not determine initial_sigma.")
        return float("inf")


def qsilk_micrograin(
    x: torch.Tensor,
    q_low: float = 0.001,
    q_high: float = 0.999,
    alpha: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if x.dim() != 4:
        return x

    original_dtype = x.dtype
    x_float = x.float()
    batch_size = x_float.shape[0]
    flattened = x_float.view(batch_size, -1)

    low = torch.quantile(flattened, q_low, dim=1)
    high = torch.quantile(flattened, q_high, dim=1)

    mid = (low + high) * 0.5
    delta = (high - low) * 0.5

    mid = mid.view(batch_size, 1, 1, 1)
    delta = delta.view(batch_size, 1, 1, 1)

    normed = (x_float - mid) / (delta + eps)
    stabilized = mid + delta * torch.tanh(alpha * normed)
    return stabilized.to(dtype=original_dtype)


def ndtri(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q_clamped = torch.clamp(q, eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(q_clamped * 2.0 - 1.0)


def qsilk_aqclip_lite(
    x: torch.Tensor,
    tile_size: int = 32,
    stride: int = 16,
    alpha: float = 2.0,
    ema_state: Dict[str, torch.Tensor] | None = None,
    ema_beta: float = 0.8,
    eps: float = 1e-8,
):
    if ema_state is None:
        ema_state = {}
    if x.dim() != 4:
        return x, ema_state

    original_dtype = x.dtype
    x_float = x.float()
    b, c, h, w = x_float.shape

    z_m = x_float.mean(dim=1, keepdim=True)

    g_x = z_m[:, :, :, 2:] - z_m[:, :, :, :-2]
    g_y = z_m[:, :, 2:, :] - z_m[:, :, :-2, :]
    g_x = F.pad(g_x, (1, 1, 0, 0))
    g_y = F.pad(g_y, (0, 0, 1, 1))
    g = torch.sqrt(g_x ** 2 + g_y ** 2)

    g_abs = g.abs()
    g_max = g_abs.amax(dim=(2, 3), keepdim=True)
    if torch.all(g_max <= eps):
        return x, ema_state

    g_norm = g_abs / (g_max + eps)

    g_grid = F.avg_pool2d(g_norm, kernel_size=tile_size, stride=stride)
    q_low_tile = 0.5 * (g_grid ** 2)
    q_high_tile = 1.0 - 0.5 * ((1.0 - g_grid) ** 2)

    unfold = torch.nn.Unfold(kernel_size=tile_size, stride=stride)
    fold = torch.nn.Fold(output_size=(h, w), kernel_size=tile_size, stride=stride)

    x_flat = x_float.view(b * c, 1, h, w)
    patches = unfold(x_flat)
    mu = patches.mean(dim=1, keepdim=True)
    sigma = patches.std(dim=1, keepdim=True) + eps

    q_low_flat = q_low_tile.view(b, 1, -1)
    q_high_flat = q_high_tile.view(b, 1, -1)
    q_low_bc = q_low_flat.repeat_interleave(c, dim=0)
    q_high_bc = q_high_flat.repeat_interleave(c, dim=0)

    low_corridor = mu + sigma * ndtri(q_low_bc, eps)
    high_corridor = mu + sigma * ndtri(q_high_bc, eps)

    if "l" not in ema_state or ema_state["l"].shape != low_corridor.shape:
        ema_state["l"] = low_corridor.detach()
        ema_state["h"] = high_corridor.detach()
    else:
        ema_state["l"] = ema_beta * ema_state["l"] + (1.0 - ema_beta) * low_corridor.detach()
        ema_state["h"] = ema_beta * ema_state["h"] + (1.0 - ema_beta) * high_corridor.detach()

    l_corridor = ema_state["l"]
    h_corridor = ema_state["h"]

    m = 0.5 * (l_corridor + h_corridor)
    delta = 0.5 * (h_corridor - l_corridor)
    normed = (patches - m) / (delta + eps)
    patches_clipped = m + delta * torch.tanh(alpha * normed)

    x_clipped_flat = fold(patches_clipped)
    norm = fold(unfold(torch.ones_like(x_flat)))
    x_clipped_flat = x_clipped_flat / (norm + eps)
    x_clipped = x_clipped_flat.view(b, c, h, w)

    return x_clipped.to(dtype=original_dtype), ema_state


def make_cfg_zero_base_builder(zero_init_first_step: bool, initial_sigma: float) -> Callable:
    def builder(args) -> GuidanceState:
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]

        if zero_init_first_step:
            current_sigma_val = args["sigma"][0].item()
            if abs(current_sigma_val - initial_sigma) < 1e-5:
                zero_tensor = uncond_denoised * 0.0
                return GuidanceState(zero_tensor, zero_tensor)

        original_shape = cond_denoised.shape
        batch_size = original_shape[0] if len(original_shape) > 0 else 1
        positive_flat = cond_denoised.reshape(batch_size, -1)
        negative_flat = uncond_denoised.reshape(batch_size, -1)
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm_negative = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        st_star = dot_product / squared_norm_negative
        st_star_reshaped = st_star.view(batch_size, *([1] * (len(original_shape) - 1)))
        base_pred = uncond_denoised * st_star_reshaped
        guidance_direction = cond_denoised - base_pred
        guidance_term = guidance_direction * cond_scale
        return GuidanceState(base_pred, guidance_term)

    return builder


def make_zeresfdg_base_builder() -> Callable:
    def builder(args) -> GuidanceState:
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]

        original_shape = cond_denoised.shape
        batch_size = original_shape[0] if len(original_shape) > 0 else 1
        positive_flat = cond_denoised.reshape(batch_size, -1)
        negative_flat = uncond_denoised.reshape(batch_size, -1)
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm_negative = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        alpha_parallel = dot_product / squared_norm_negative
        alpha_parallel = alpha_parallel.view(batch_size, *([1] * (len(original_shape) - 1)))
        base_pred = uncond_denoised * alpha_parallel
        residual = cond_denoised - base_pred
        guidance_term = residual * cond_scale
        return GuidanceState(base_pred, guidance_term)

    return builder


def make_fdg_modifier(w_low: float, w_high: float, fdg_levels: int) -> Callable:
    def modifier(args, state: GuidanceState) -> GuidanceState:
        cond_scale = args["cond_scale"]
        if cond_scale != 0:
            guidance_direction = state.guidance_term / cond_scale
        else:
            guidance_direction = state.guidance_term

        try:
            guidance_low_freq_scaled = guidance_direction * w_low
            guidance_high_freq_scaled = guidance_direction * w_high
            levels = max(2, int(fdg_levels))
            low_freq_part_from_low = build_laplacian_pyramid(guidance_low_freq_scaled, levels)[-1]
            low_freq_part_from_high = build_laplacian_pyramid(guidance_high_freq_scaled, levels)[-1]

            if low_freq_part_from_high.shape != guidance_high_freq_scaled.shape:
                low_freq_part_from_high = torch.nn.functional.interpolate(
                    low_freq_part_from_high,
                    size=guidance_high_freq_scaled.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if low_freq_part_from_low.shape != guidance_high_freq_scaled.shape:
                low_freq_part_from_low = torch.nn.functional.interpolate(
                    low_freq_part_from_low,
                    size=guidance_high_freq_scaled.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            high_freq_part = guidance_high_freq_scaled - low_freq_part_from_high
            final_guidance_term = (high_freq_part + low_freq_part_from_low) * cond_scale
        except Exception as e:
            logging.error(f"FDG: Error during frequency blending: {e}. Falling back to standard guidance.")
            final_guidance_term = state.guidance_term

        return GuidanceState(state.base_prediction, final_guidance_term)

    return modifier


def make_zeresfdg_modifier(
    w_low: float,
    w_high: float,
    alpha: float,
    tau_lo: float,
    tau_hi: float,
    beta: float,
    controller_enabled: bool,
    sigma: float = 1.0,
):
    eps = 1e-8
    rho = None
    mode_state = None

    def gaussian_lowpass(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 4:
            return tensor
        kernel_size = int(max(3, 2 * round(3 * sigma) + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        try:
            return gaussian_blur2d(
                tensor,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                border_type="reflect",
            )
        except Exception as e:
            logging.error(f"ZeResFDG: Gaussian blur failed with error {e}; skipping blur.")
            return tensor

    def ensure_mask(mask, target: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=target.device, dtype=target.dtype)
        else:
            mask = mask.to(device=target.device, dtype=target.dtype)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        if mask.shape[0] == 1 and target.shape[0] > 1:
            mask = mask.expand(target.shape[0], *mask.shape[1:])

        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask, size=target.shape[-2:], mode="bilinear", align_corners=False)

        if mask.shape[1] == 1 and target.shape[1] > 1:
            mask = mask.expand(target.shape[0], target.shape[1], *mask.shape[-2:])

        return mask

    def apply_mask(tensor: torch.Tensor, mask) -> torch.Tensor:
        mask_tensor = ensure_mask(mask, tensor)
        if mask_tensor is None:
            return tensor
        return tensor * mask_tensor

    def update_mode(ratio: torch.Tensor):
        nonlocal mode_state
        if not controller_enabled:
            mode_state = torch.ones_like(ratio, dtype=torch.bool)
            return mode_state

        if mode_state is None or mode_state.shape != ratio.shape:
            mode_state = ratio > tau_hi
        mode_state = torch.where(ratio > tau_hi, torch.ones_like(mode_state, dtype=torch.bool), mode_state)
        mode_state = torch.where(ratio < tau_lo, torch.zeros_like(mode_state, dtype=torch.bool), mode_state)
        return mode_state

    def modifier(args, state: GuidanceState) -> GuidanceState:
        nonlocal rho
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]
        guidance_mask = args.get("guidance_mask") or args.get("mask") or args.get("g")

        if cond_scale != 0:
            residual = state.guidance_term / cond_scale
        else:
            residual = state.guidance_term

        delta = cond_denoised - uncond_denoised
        delta_low = gaussian_lowpass(delta)
        delta_high = delta - delta_low

        batch_size = delta.shape[0]
        low_norm = torch.sum(delta_low.reshape(batch_size, -1) ** 2, dim=1)
        high_norm = torch.sum(delta_high.reshape(batch_size, -1) ** 2, dim=1)
        r_hf = high_norm / (low_norm + high_norm + eps)

        if rho is None or rho.shape != r_hf.shape:
            rho = r_hf.detach()
        else:
            rho = beta * rho + (1 - beta) * r_hf.detach()

        mode = update_mode(rho)

        residual_low = gaussian_lowpass(residual)
        residual_high = residual - residual_low
        fdg_residual = w_low * residual_low + w_high * residual_high
        fdg_residual = apply_mask(fdg_residual, guidance_mask)
        conservative_term = fdg_residual * cond_scale

        fdg_delta = w_low * delta_low + w_high * delta_high
        fdg_delta = apply_mask(fdg_delta, guidance_mask)
        y_cfg = uncond_denoised + cond_scale * fdg_delta

        dims = tuple(range(1, cond_denoised.dim()))
        target_std = cond_denoised.std(dim=dims, keepdim=True)
        y_cfg_std = y_cfg.std(dim=dims, keepdim=True)
        rescale_factor = torch.where(y_cfg_std > 0, target_std / (y_cfg_std + eps), torch.ones_like(target_std))
        y_rescaled = y_cfg * rescale_factor
        rescaled_prediction = alpha * y_rescaled + (1 - alpha) * y_cfg
        rescale_term = rescaled_prediction - state.base_prediction

        if mode.dtype != torch.bool:
            mode = mode > 0
        mode = mode.view(batch_size, *([1] * (rescale_term.dim() - 1)))
        combined_guidance = rescale_term * mode + conservative_term * (~mode)

        return GuidanceState(state.base_prediction, combined_guidance)

    return modifier


def get_prediction_with_dropped_blocks(diffusion_model, model_function, drop_ratio: float, model_kwargs):
    target_modules = []
    if hasattr(diffusion_model, "blocks"):
        target_modules = diffusion_model.blocks
    elif hasattr(diffusion_model, "output_blocks") and hasattr(diffusion_model, "input_blocks"):
        target_modules = diffusion_model.input_blocks + diffusion_model.output_blocks

    if not target_modules:
        logging.warning("S2-Guidance: Could not find target blocks to drop. S2-Guidance will have no effect.")
        return model_function(**model_kwargs)

    num_blocks_to_drop = int(len(target_modules) * drop_ratio)
    if num_blocks_to_drop == 0:
        return model_function(**model_kwargs)

    original_forwards = {}
    blocks_to_drop_indices = random.sample(range(len(target_modules)), num_blocks_to_drop)
    identity_forward = lambda x, *args, **kwargs: x

    try:
        for i in blocks_to_drop_indices:
            module = target_modules[i]
            original_forwards[i] = module.forward
            module.forward = identity_forward

        logging.debug(f"S2-Guidance: Dropped {num_blocks_to_drop} blocks.")
        prediction = model_function(**model_kwargs)
    finally:
        for i, original_forward in original_forwards.items():
            target_modules[i].forward = original_forward

    return prediction


def make_s2_modifier(model, s2_drop_ratio: float, s2_scale_omega: float) -> Callable:
    def modifier(args, state: GuidanceState) -> GuidanceState:
        model_options = args.get("model_options", {}) or {}
        cond_list = args["cond"]
        sigma = args["sigma"]
        x = args["input"]
        t = sigma

        if not (isinstance(cond_list, list) and len(cond_list) > 0 and isinstance(cond_list[0], dict)):
            logging.warning("S2-Guidance: unexpected cond format; skip S2 this step.")
            return state
        h0 = cond_list[0]

        c_crossattn_tensor = h0.get("cross_attn", None)
        if c_crossattn_tensor is None:
            logging.warning("S2-Guidance: cross_attn tensor not found; skip S2 this step.")
            return state

        needs_y = getattr(model.model.diffusion_model, "num_classes", None) is not None
        y_tensor = h0.get("pooled_output", None) if needs_y else None
        if needs_y and y_tensor is None:
            logging.warning("S2-Guidance: model needs 'y' but pooled_output not found; skip S2 this step.")
            return state

        c_concat = None
        mc = h0.get("model_conds", None)
        if isinstance(mc, dict) and "c_concat" in mc:
            v = mc["c_concat"]
            c_concat = getattr(v, "cond", v)

        control = h0.get("control", None)

        model_kwargs = {
            "x": x,
            "t": t,
            "c_crossattn": c_crossattn_tensor,
            "transformer_options": model_options,
        }
        if needs_y:
            model_kwargs["y"] = y_tensor
        if c_concat is not None:
            model_kwargs["c_concat"] = c_concat
        if control is not None:
            model_kwargs["control"] = control

        denoised_s = get_prediction_with_dropped_blocks(
            model.model.diffusion_model,
            model.model.apply_model,
            s2_drop_ratio,
            model_kwargs,
        )
        new_prediction = state.prediction - s2_scale_omega * denoised_s
        zero_term = torch.zeros_like(state.guidance_term)
        return GuidanceState(new_prediction, zero_term)

    return modifier


def make_qsilk_modifier(
    micro_q_low: float = 0.001,
    micro_q_high: float = 0.999,
    micro_alpha: float = 2.0,
    use_aqclip: bool = True,
    tile_size: int = 32,
    stride: int = 16,
    aqclip_alpha: float = 2.0,
    ema_beta: float = 0.8,
) -> Callable:
    ema_state: Dict[str, torch.Tensor] = {}

    def modifier(args, state: GuidanceState) -> GuidanceState:
        nonlocal ema_state

        pred = qsilk_micrograin(
            state.prediction,
            q_low=micro_q_low,
            q_high=micro_q_high,
            alpha=micro_alpha,
        )

        if use_aqclip:
            pred, ema_state = qsilk_aqclip_lite(
                pred,
                tile_size=tile_size,
                stride=stride,
                alpha=aqclip_alpha,
                ema_state=ema_state,
                ema_beta=ema_beta,
            )

        guidance_term = pred - state.base_prediction
        return GuidanceState(state.base_prediction, guidance_term)

    return modifier


def enable_qsilk_modifier(
    model,
    micro_q_low: float = 0.001,
    micro_q_high: float = 0.999,
    micro_alpha: float = 2.0,
    use_aqclip: bool = True,
    tile_size: int = 32,
    stride: int = 16,
    aqclip_alpha: float = 2.0,
    ema_beta: float = 0.8,
):
    """Attach the QSilk modifier to the model's guidance pipeline.

    This keeps QSilk purely code-configured while ensuring it executes last in
    the pipeline order without adding any UI/node bindings.
    """

    pipeline = ensure_guidance_pipeline(model)
    modifier = make_qsilk_modifier(
        micro_q_low=micro_q_low,
        micro_q_high=micro_q_high,
        micro_alpha=micro_alpha,
        use_aqclip=use_aqclip,
        tile_size=tile_size,
        stride=stride,
        aqclip_alpha=aqclip_alpha,
        ema_beta=ema_beta,
    )

    # Preserve insertion order by re-inserting to keep QSilk last.
    if "qsilk" in pipeline.modifiers:
        pipeline.modifiers.pop("qsilk")
    pipeline.add_modifier("qsilk", modifier)

    return pipeline


GUIDANCE_PARAM_KEYS = [
    "CFG-Zero Enabled",
    "CFG-Zero Init First Step",
    "FDG Enabled",
    "FDG w_low",
    "FDG w_high",
    "FDG Levels",
    "ZeResFDG Enabled",
    "ZeResFDG λ_l",
    "ZeResFDG λ_h",
    "ZeResFDG α",
    "ZeResFDG τ_lo",
    "ZeResFDG τ_hi",
    "ZeResFDG β",
    "ZeResFDG Controller",
    "S2-Guidance Enabled",
    "S2-Guidance Omega",
    "S2-Guidance Drop Ratio",
]


def reset_unet_if_needed(p) -> bool:
    if not hasattr(p, "_guidance_original_unet"):
        p._guidance_original_unet = p.sd_model.forge_objects.unet.clone()
    if getattr(p, "_guidance_unet_restored", False):
        return False
    p.sd_model.forge_objects.unet = p._guidance_original_unet.clone()
    p._guidance_unet_restored = True
    return True


def clear_generation_params(p, keys):
    for param in keys:
        p.extra_generation_params.pop(param, None)


def clear_generation_params_once(p, keys=GUIDANCE_PARAM_KEYS):
    if getattr(p, "_guidance_params_cleared", False):
        return False
    clear_generation_params(p, keys)
    p._guidance_params_cleared = True
    return True

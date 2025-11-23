import logging
import random
from typing import Callable, Dict

import torch
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


GUIDANCE_PARAM_KEYS = [
    "CFG-Zero Enabled",
    "CFG-Zero Init First Step",
    "FDG Enabled",
    "FDG w_low",
    "FDG w_high",
    "FDG Levels",
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

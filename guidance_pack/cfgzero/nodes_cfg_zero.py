# --- START OF FILE nodes_cfg_zero (4).py ---
import torch
import logging
import kornia
import random
from kornia.geometry.transform import build_laplacian_pyramid

def build_image_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        upsampled = torch.nn.functional.interpolate(img, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
        img = upsampled + pyramid[i]
    return img

class CFGZeroNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_zero_enabled": ("BOOLEAN", {"default": False}),
                "zero_init_first_step": ("BOOLEAN", {"default": False}),
                "fdg_enabled": ("BOOLEAN", {"default": False}),
                "w_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "w_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "fdg_levels": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
                "s2_guidance_enabled": ("BOOLEAN", {"default": False}),
                "s2_scale_omega": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.01}),
                "s2_drop_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "round": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies CFG-Zero, FDG, and/or S2-Guidance scaling."

    def patch(self, model, mahiro_enabled: bool = False, cfg_zero_enabled: bool = False, zero_init_first_step: bool = False,
              fdg_enabled: bool = False, w_low: float = 1.0, w_high: float = 1.0, fdg_levels: int = 3,
              s2_guidance_enabled: bool = False, s2_scale_omega: float = 0.25, s2_drop_ratio: float = 0.1):

        if not cfg_zero_enabled and not fdg_enabled and not s2_guidance_enabled:
            return (model,)

        m = model.clone()

        try:
            initial_sigma = m.model.model_sampling.sigma_max
        except AttributeError:
            logging.warning("Custom Guidance: Could not determine initial_sigma.")
            initial_sigma = float('inf')

        def get_prediction_with_dropped_blocks(model_function, model_kwargs):
            target_modules = []
            if hasattr(m.model.diffusion_model, 'blocks'):
                target_modules = m.model.diffusion_model.blocks
            elif hasattr(m.model.diffusion_model, 'output_blocks') and hasattr(m.model.diffusion_model, 'input_blocks'):
                target_modules = m.model.diffusion_model.input_blocks + m.model.diffusion_model.output_blocks
            
            if not target_modules:
                logging.warning("S2-Guidance: Could not find target blocks to drop. S2-Guidance will have no effect.")
                return model_function(**model_kwargs)

            num_blocks_to_drop = int(len(target_modules) * s2_drop_ratio)
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

        def guidance_function(args):
            cond_scale = args['cond_scale']
            cond_denoised = args['cond_denoised']
            uncond_denoised = args['uncond_denoised']
            
            if cfg_zero_enabled and zero_init_first_step:
                current_sigma_val = args['sigma'][0].item()
                if abs(current_sigma_val - initial_sigma) < 1e-5:
                    return uncond_denoised * 0.0

            if cfg_zero_enabled:
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
            else:
                base_pred = uncond_denoised
                guidance_direction = cond_denoised - uncond_denoised

            if fdg_enabled:
                logging.debug(f"FDG: Applying w_low={w_low}, w_high={w_high}")
                try:
                    guidance_low_freq_scaled = guidance_direction * w_low
                    guidance_high_freq_scaled = guidance_direction * w_high
                    levels = max(2, int(fdg_levels))
                    low_freq_part_from_low = build_laplacian_pyramid(guidance_low_freq_scaled, levels)[-1]
                    low_freq_part_from_high = build_laplacian_pyramid(guidance_high_freq_scaled, levels)[-1]
                    
                    if low_freq_part_from_high.shape != guidance_high_freq_scaled.shape:
                         low_freq_part_from_high = torch.nn.functional.interpolate(low_freq_part_from_high, size=guidance_high_freq_scaled.shape[-2:], mode='bilinear', align_corners=False)
                    if low_freq_part_from_low.shape != guidance_high_freq_scaled.shape:
                         low_freq_part_from_low = torch.nn.functional.interpolate(low_freq_part_from_low, size=guidance_high_freq_scaled.shape[-2:], mode='bilinear', align_corners=False)

                    high_freq_part = guidance_high_freq_scaled - low_freq_part_from_high
                    final_guidance_term = (high_freq_part + low_freq_part_from_low) * cond_scale
                except Exception as e:
                    logging.error(f"FDG: Error during frequency blending: {e}. Falling back to standard guidance.")
                    final_guidance_term = guidance_direction * cond_scale
            else:
                final_guidance_term = guidance_direction * cond_scale
            
            combined_pred = base_pred + final_guidance_term

            if s2_guidance_enabled:
                # --- S2 for Forge/Comfy (Test-reForge) ---
                model_options = args.get('model_options', {}) or {}
                cond_list = args['cond']
                sigma = args['sigma']
                x = args['input']
                t = sigma

                # lấy item cond đầu tiên (đủ cho S² branch)
                if not (isinstance(cond_list, list) and len(cond_list) > 0 and isinstance(cond_list[0], dict)):
                    logging.warning("S2-Guidance: unexpected cond format; skip S2 this step.")
        
            # --- Mahiro gating (optional) -------------------------------------
            if mahiro_enabled:
                scale = cond_scale
                C = cond_denoised
                U = uncond_denoised
                leap = C * scale
                # blend our improved guidance (combined_pred) with pure-cond leap
                merge = 0.5 * (leap + combined_pred)

                def srs(x: torch.Tensor):
                    return torch.sqrt(x.abs() + 1e-12) * x.sign()

                u_leap = U * scale
                # cosine similarity along channel/spatial dims; average across batch
                sim = torch.nn.functional.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, U.ndim))).mean()
                simsc = 2.0 * (sim + 1.0)  # [0,4]
                combined_pred = (simsc * combined_pred + (4.0 - simsc) * leap) / 4.0

            return combined_pred
                h0 = cond_list[0]

                # cross-attn THẬT (tensor), không dùng wrapper
                c_crossattn_tensor = h0.get("cross_attn", None)
                if c_crossattn_tensor is None:
                    logging.warning("S2-Guidance: cross_attn tensor not found; skip S2 this step.")
        
            # --- Mahiro gating (optional) -------------------------------------
            if mahiro_enabled:
                scale = cond_scale
                C = cond_denoised
                U = uncond_denoised
                leap = C * scale
                # blend our improved guidance (combined_pred) with pure-cond leap
                merge = 0.5 * (leap + combined_pred)

                def srs(x: torch.Tensor):
                    return torch.sqrt(x.abs() + 1e-12) * x.sign()

                u_leap = U * scale
                # cosine similarity along channel/spatial dims; average across batch
                sim = torch.nn.functional.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, U.ndim))).mean()
                simsc = 2.0 * (sim + 1.0)  # [0,4]
                combined_pred = (simsc * combined_pred + (4.0 - simsc) * leap) / 4.0

            return combined_pred

                # nếu UNet là class-conditional thì bắt buộc có y
                needs_y = getattr(m.model.diffusion_model, "num_classes", None) is not None
                y_tensor = h0.get("pooled_output", None) if needs_y else None
                if needs_y and y_tensor is None:
                    logging.warning("S2-Guidance: model needs 'y' but pooled_output not found; skip S2 this step.")
        
            # --- Mahiro gating (optional) -------------------------------------
            if mahiro_enabled:
                scale = cond_scale
                C = cond_denoised
                U = uncond_denoised
                leap = C * scale
                # blend our improved guidance (combined_pred) with pure-cond leap
                merge = 0.5 * (leap + combined_pred)

                def srs(x: torch.Tensor):
                    return torch.sqrt(x.abs() + 1e-12) * x.sign()

                u_leap = U * scale
                # cosine similarity along channel/spatial dims; average across batch
                sim = torch.nn.functional.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, U.ndim))).mean()
                simsc = 2.0 * (sim + 1.0)  # [0,4]
                combined_pred = (simsc * combined_pred + (4.0 - simsc) * leap) / 4.0

            return combined_pred

                # c_concat (nếu có) nằm trong model_conds dưới dạng wrapper CONDRegular
                c_concat = None
                mc = h0.get("model_conds", None)
                if isinstance(mc, dict) and "c_concat" in mc:
                    v = mc["c_concat"]
                    c_concat = getattr(v, "cond", v)  # lấy tensor từ wrapper

                # control (nếu có) được forge_sampler đẩy vào mỗi h
                control = h0.get("control", None)

                model_kwargs = {
                    "x": x,
                    "t": t,
                    "c_crossattn": c_crossattn_tensor,
                    "transformer_options": model_options,  # tên đúng trong Forge
                }
                if needs_y:
                    model_kwargs["y"] = y_tensor
                if c_concat is not None:
                    model_kwargs["c_concat"] = c_concat
                if control is not None:
                    model_kwargs["control"] = control

                denoised_s = get_prediction_with_dropped_blocks(m.model.apply_model, model_kwargs)
                combined_pred = combined_pred - s2_scale_omega * denoised_s


            # --- Mahiro gating (optional) -------------------------------------
            if mahiro_enabled:
                scale = cond_scale
                C = cond_denoised
                U = uncond_denoised
                leap = C * scale
                # blend our improved guidance (combined_pred) with pure-cond leap
                merge = 0.5 * (leap + combined_pred)

                def srs(x: torch.Tensor):
                    return torch.sqrt(x.abs() + 1e-12) * x.sign()

                u_leap = U * scale
                # cosine similarity along channel/spatial dims; average across batch
                sim = torch.nn.functional.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, U.ndim))).mean()
                simsc = 2.0 * (sim + 1.0)  # [0,4]
                combined_pred = (simsc * combined_pred + (4.0 - simsc) * leap) / 4.0

            return combined_pred

        m.set_model_sampler_post_cfg_function(guidance_function, "custom_guidance")
        logging.debug(f"Patched model with CFG-Zero: {cfg_zero_enabled}, FDG: {fdg_enabled}, S2-Guidance: {s2_guidance_enabled}")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "CFGZeroFDGS2Node": CFGZeroNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGZeroFDGS2Node": "YX-CFG-Zero/FDG/S2 Guidance"
}
# --- END OF FILE nodes_cfg_zero (4).py ---

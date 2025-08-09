# --- START OF FILE nodes_cfg_zero.py ---
import torch
import logging
import kornia
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
                # --- Tham số cho CFG-Zero ---
                "cfg_zero_enabled": ("BOOLEAN", {"default": False}),
                "zero_init_first_step": ("BOOLEAN", {"default": False}),
                # --- Tham số cho FDG ---
                "fdg_enabled": ("BOOLEAN", {"default": False}),
                "w_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "w_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "fdg_levels": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies CFG-Zero and/or FDG guidance scaling."

    def patch(self, model, cfg_zero_enabled: bool = False, zero_init_first_step: bool = False, 
              fdg_enabled: bool = False, w_low: float = 1.0, w_high: float = 1.0, fdg_levels: int = 3):
        
        # Nếu không có tính năng nào được bật, không cần patch mô hình
        if not cfg_zero_enabled and not fdg_enabled:
            return (model,) # Trả về mô hình gốc

        m = model.clone()
        
        try:
            initial_sigma = m.model.model_sampling.sigma_max 
        except AttributeError:
            logging.warning("CFG-Zero/FDG: Could not determine initial_sigma.")
            initial_sigma = float('inf')

        def guidance_function(args):
            cond_scale = args['cond_scale']
            cond_denoised = args['cond_denoised']
            uncond_denoised = args['uncond_denoised']

            # --- Logic Zero Init (chỉ hoạt động khi CFG-Zero bật) ---
            if cfg_zero_enabled and zero_init_first_step:
                current_sigma_val = args['sigma'][0].item()
                if abs(current_sigma_val - initial_sigma) < 1e-5:
                    return uncond_denoised * 0.0

            # --- Bước 1: Xác định dự đoán cơ sở (base_pred) và hướng điều hướng (guidance_direction) ---
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
                # Nếu CFG-Zero tắt, sử dụng CFG chuẩn
                base_pred = uncond_denoised
                guidance_direction = cond_denoised - uncond_denoised

            # --- Bước 2: Áp dụng FDG nếu được bật ---
            if fdg_enabled:
                logging.debug(f"FDG: Applying w_low={w_low}, w_high={w_high}")
                try:
                    # Tính toán hai phiên bản của guidance term
                    guidance_low_freq_scaled = guidance_direction * w_low
                    guidance_high_freq_scaled = guidance_direction * w_high

                    # Lấy thành phần tần số thấp từ mỗi phiên bản
                    # Chúng ta chỉ cần thành phần tần số thấp (residual) của pyramid
                    levels = max(2, int(fdg_levels))
                    
                    low_freq_part_from_low = build_laplacian_pyramid(guidance_low_freq_scaled, levels)[-1]
                    low_freq_part_from_high = build_laplacian_pyramid(guidance_high_freq_scaled, levels)[-1]
                    
                    # Phiên bản "high_freq_scaled" sẽ là nền, và chúng ta sẽ thay thế phần tần số thấp của nó
                    # bằng phần tần số thấp đã được scale bởi w_low.
                    #
                    # final = (guidance_high_freq_scaled - low_freq_part_from_high) + low_freq_part_from_low
                    #   ^-- đây là thành phần tần số cao của phiên bản high
                    #                                                               ^-- đây là thành phần tần số thấp của phiên bản low
                    # Phép toán này bảo toàn kích thước vì không có up/down sampling và tái tạo.
                    
                    # QUICK FIX CHO LỖI KÍCH THƯỚC TRONG PYRAMID
                    # Đảm bảo các thành phần tần số thấp có kích thước khớp nhau trước khi trừ/cộng
                    if low_freq_part_from_high.shape != guidance_high_freq_scaled.shape:
                         low_freq_part_from_high = torch.nn.functional.interpolate(low_freq_part_from_high, size=guidance_high_freq_scaled.shape[-2:], mode='bilinear', align_corners=False)
                    
                    if low_freq_part_from_low.shape != guidance_high_freq_scaled.shape:
                         low_freq_part_from_low = torch.nn.functional.interpolate(low_freq_part_from_low, size=guidance_high_freq_scaled.shape[-2:], mode='bilinear', align_corners=False)

                    # Trộn chúng lại với nhau
                    high_freq_part = guidance_high_freq_scaled - low_freq_part_from_high
                    final_guidance_term = (high_freq_part + low_freq_part_from_low) * cond_scale
                    
                except Exception as e:
                    logging.error(f"FDG: Error during frequency blending: {e}. Falling back to standard guidance.")
                    final_guidance_term = guidance_direction * cond_scale
            else:
                # Nếu FDG tắt, sử dụng CFG scale chuẩn
                final_guidance_term = guidance_direction * cond_scale
            # --- Bước 3: Kết hợp lại ---
            combined_pred = base_pred + final_guidance_term
            
            return combined_pred

        m.set_model_sampler_post_cfg_function(guidance_function, "cfg_zero_fdg_guidance")
        logging.debug(f"Patched model with CFG-Zero: {cfg_zero_enabled}, FDG: {fdg_enabled}")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "CFGZeroFDGNode": CFGZeroNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGZeroFDGNode": "YX-CFG-Zero/FDG Guidance"
}
# --- END OF FILE nodes_cfg_zero.py ---

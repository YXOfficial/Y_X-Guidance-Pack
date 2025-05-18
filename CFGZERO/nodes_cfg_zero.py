# --- START OF FILE nodes_cfg_zero.py ---
import torch
import logging

class CFGZeroNode:
    @classmethod
    def INPUT_TYPES(s): # Not strictly needed for WebUI script use, but good practice
        return {
            "required": {
                "model": ("MODEL",),
                "zero_init_first_step": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches" # Or similar
    DESCRIPTION = "Applies CFG-Zero guidance scaling."

    def patch(self, model, zero_init_first_step: bool = False):
        m = model.clone()
        
        # Capture initial sigma for first step detection
        # sigma_max is the sigma for the first step (highest noise)
        try:
            # For UNet models (typical SD)
            initial_sigma = m.model.model_sampling.sigma_max 
        except AttributeError:
            # Fallback or error if model structure is different (e.g. VAE, CLIP)
            logging.warning("CFG-Zero: Could not determine initial_sigma from model. First step zero-init might not work correctly.")
            initial_sigma = float('inf') # Make first step detection unlikely if sigma_max is not found

        # This is the function that will be called by the sampler
        def cfg_zero_guidance_function(args):
            cond_scale = args['cond_scale']
            cond_denoised = args['cond_denoised']     # Corresponds to noise_pred_text
            uncond_denoised = args['uncond_denoised'] # Corresponds to noise_pred_uncond
            # sigma_t = args['sigma'] # Current sigma for this step (tensor)
            # x = args['x'] # Current latent

            # First step zero-init logic
            if zero_init_first_step:
                current_sigma_val = args['sigma'][0].item() # Get scalar value
                # Compare with a small tolerance due to potential float precision issues
                if abs(current_sigma_val - initial_sigma) < 1e-5:
                    logging.debug(f"CFG-Zero: Applying zero_init for first step (sigma: {current_sigma_val})")
                    return uncond_denoised * 0.0 # Perform zero init

            # Reshape for broadcasting if necessary, but usually not needed as operations are element-wise
            # Original shapes are typically (batch_size, channels, height, width)
            
            # Keep batch dimension, flatten others for st_star calculation
            # Ensure original_shape and batch_size are correctly derived
            original_shape = cond_denoised.shape
            if len(original_shape) == 0: # Should not happen with tensors
                return uncond_denoised + cond_scale * (cond_denoised - uncond_denoised) # Fallback to standard CFG

            batch_size = original_shape[0] if len(original_shape) > 0 else 1

            # Flatten starting from the first dimension (channels)
            # (B, C, H, W) -> (B, C*H*W)
            positive_flat = cond_denoised.reshape(batch_size, -1)
            negative_flat = uncond_denoised.reshape(batch_size, -1)

            # Calculate st_star = (v_cond^T * v_uncond) / ||v_uncond||^2
            # dot_product: sum over the flattened dimensions for each item in batch
            dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
            
            # squared_norm_negative: sum of squares over flattened dimensions for each item in batch
            squared_norm_negative = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8 # Epsilon for stability

            st_star = dot_product / squared_norm_negative
            
            # Reshape st_star for broadcasting with original tensor shapes
            # (batch_size, 1) -> (batch_size, 1, 1, 1) for 4D tensors like (B,C,H,W)
            # This ensures st_star applies correctly element-wise during multiplication
            st_star_reshaped = st_star.view(batch_size, *([1] * (len(original_shape) - 1)))

            # CFG-Zero formula:
            # noise_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_text - noise_pred_uncond * st_star)
            # noise_pred = uncond_denoised * st_star_reshaped + \
            #              cond_scale * (cond_denoised - uncond_denoised * st_star_reshaped)
            
            # Alternative interpretation from some discussions (aligns more with scaling the "difference" part):
            # This is essentially: uncond + st_star * scale * (cond - uncond) - this seems less likely from original paper
            # The provided snippet is: uncond_p * st_star + scale * (cond_p - uncond_p * st_star)
            # Let's stick to the snippet's formula.

            combined_pred = uncond_denoised * st_star_reshaped + \
                            cond_scale * (cond_denoised - uncond_denoised * st_star_reshaped)
            
            return combined_pred

        # "cfg_zero_guidance" is a unique name for this specific patch
        m.set_model_sampler_post_cfg_function(cfg_zero_guidance_function, "cfg_zero_guidance")
        logging.debug(f"CFG-Zero: Model patched with zero_init_first_step = {zero_init_first_step}, initial_sigma_captured = {initial_sigma}")
        return (m,)

# For ComfyUI compatibility (optional, but good practice)
NODE_CLASS_MAPPINGS = {
    "CFGZeroNode": CFGZeroNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGZeroNode": "YX-CFG-Zero Guidance Patcher"
}

# --- END OF FILE nodes_cfg_zero.py ---

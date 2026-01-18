import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules import devices

class TPSONode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "p": ("PROCESSING",),
                "tpso_enabled": ("BOOLEAN", {"default": False}),
                "tpso_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "tpso_lr": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "tpso_lambda": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tpso_r": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies Token-Prompt embedding Space Optimization (TPSO) via ModelPatcher."

    def patch(self, unet, p, tpso_enabled=False, tpso_steps=10, tpso_lr=0.01, tpso_lambda=0.5, tpso_r=0.4, tpso_kappa=0.8):
        if not tpso_enabled:
            return (unet,)

        logging.info(f"TPSO: Starting optimization for {len(p.prompts)} prompts...")
        
        # 1. Optimization Phase (Context Optimization)
        # We optimize the embedding (cond) directly since we can't easily backprop through CLIP in all environments.
        
        # Get original embeddings
        # We use p.sd_model to get conditioning.
        # This gives us the target to stay close to.
        with torch.no_grad():
            original_cond = p.sd_model.get_learned_conditioning(p.prompts)
            # original_cond: [B, 77, 768] (SD1.5)
        
        device = devices.device
        dtype = devices.dtype_unet
        
        # Prepare for optimization
        # We clone and detach to create a leaf tensor that requires grad
        optimized_cond = original_cond.clone().to(device, dtype=torch.float32).detach()
        optimized_cond.requires_grad_(True)
        
        optimizer = optim.Adam([optimized_cond], lr=tpso_lr)
        target_cond = original_cond.clone().to(device, dtype=torch.float32).detach()
        
        batch_size = original_cond.shape[0]
        
        # Optimization Loop
        if tpso_enabled:
            # If batch > 1, we enforce diversity between samples.
            # If batch == 1, we apply random perturbation to escape mode (simulated diversity).
            
            if batch_size > 1:
                logging.info(f"TPSO: Optimizing batch of {batch_size} for diversity...")
                for step in range(tpso_steps):
                    optimizer.zero_grad()
                    
                    # Semantic Loss: MSE to original
                    l_sem = F.mse_loss(optimized_cond, target_cond)
                    
                    # Diversity Loss: Minimize pairwise cosine similarity
                    flat = optimized_cond.view(batch_size, -1)
                    flat_norm = F.normalize(flat, p=2, dim=1)
                    similarity_matrix = torch.mm(flat_norm, flat_norm.t()) # [B, B]
                    
                    # Minimize off-diagonal
                    mask = torch.eye(batch_size, device=device).bool()
                    off_diag = similarity_matrix[~mask]
                    l_div = off_diag.mean() if off_diag.numel() > 0 else torch.tensor(0.0).to(device)
                    
                    loss = l_sem + tpso_lambda * l_div
                    
                    loss.backward()
                    optimizer.step()
                    
                logging.debug(f"TPSO: Optimization done. Final Loss: {loss.item():.4f}")
            else:
                # Single batch: Perturb to create a "variant"
                logging.info("TPSO: Single batch. Applying perturbation to escape mode.")
                with torch.no_grad():
                     # Simple random walk away from mode
                     noise = torch.randn_like(original_cond) * (tpso_lr * 10.0) 
                     optimized_cond = original_cond + noise

        # Final optimized embeddings
        final_optimized_cond = optimized_cond.to(dtype).detach()
        
        # 2. Define Wrapper for UNet Forward
        # unet is a ModelPatcher. We use set_model_unet_function_wrapper.
        
        def unet_wrapper(apply_model, args):
            # args: {'input': x, 'timestep': t, 'c': dict, ...}
            
            t = args["timestep"]
            # t might be tensor or float
            t_curr = t[0].item() if torch.is_tensor(t) else t
            
            # Progressive Schedule
            # T approx 1000.
            # If t > threshold (early steps), use optimized.
            # If t <= threshold (late steps), use original (implicit in args['c'] if we don't touch it? 
            # Wait, args['c'] passed here comes from the Sampler. 
            # The Sampler uses the prompt passed to it.
            # Usually that IS 'original_cond'.
            
            max_t = 1000.0
            threshold = max_t * (1.0 - tpso_r)
            
            if t_curr > threshold:
                # Inject Optimized Embeddings
                
                # Copy args to avoid side effects
                new_args = args.copy()
                c = args["c"]
                new_c = c.copy()
                
                if "c_crossattn" in c:
                    current_emb = c["c_crossattn"] # [Batch_Real, 77, 768]
                    
                    # Identify where to inject.
                    # Usually [Uncond, Cond] (Negative, Positive)
                    # Batch sizes:
                    # current_emb.shape[0] usually equals 2 * batch_size (if CFG > 1)
                    
                    curr_bs = current_emb.shape[0]
                    target_bs = final_optimized_cond.shape[0]
                    
                    new_emb = current_emb.clone()
                    
                    if curr_bs == 2 * target_bs:
                        # Assume [Uncond, Cond] -> Replace second half
                        new_emb[target_bs:] = final_optimized_cond
                    elif curr_bs == target_bs:
                        # Assume [Cond] (Positive only) or we are in a specific pass
                        # Replace all
                        new_emb = final_optimized_cond
                    else:
                        # Mismatch, skip to avoid error
                        pass
                    
                    new_c["c_crossattn"] = new_emb
                    new_args["c"] = new_c
                    return apply_model(new_args)
            
            # Late steps or no change -> Original
            return apply_model(args)

        # Apply the wrapper
        # We clone the patcher so we don't affect other nodes using the same model instance permanently
        patched_unet = unet.clone()
        patched_unet.set_model_unet_function_wrapper(unet_wrapper)
        
        logging.info(f"TPSO: Patch applied. Schedule: t > {1000*(1-tpso_r):.0f} uses optimized embeddings.")
        return (patched_unet,)

NODE_CLASS_MAPPINGS = {
    "TPSONode": TPSONode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TPSONode": "TPSO Guidance",
}
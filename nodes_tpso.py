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
        with torch.no_grad():
            original_cond_obj = p.sd_model.get_learned_conditioning(p.prompts)
        
        # Handle Dictionary vs Tensor (SDXL vs SD1.5)
        original_cond = original_cond_obj
        target_key = None
        
        if isinstance(original_cond_obj, dict):
            if 'crossattn' in original_cond_obj:
                target_key = 'crossattn'
            elif 'c_crossattn' in original_cond_obj:
                target_key = 'c_crossattn'
            elif 'cond' in original_cond_obj:
                 target_key = 'cond'
            else:
                target_key = next((k for k, v in original_cond_obj.items() if isinstance(v, torch.Tensor) and v.ndim == 3), None)
            
            if target_key is None:
                logging.warning("TPSO: Could not find embedding tensor in conditioning dict. Skipping TPSO.")
                return (unet,)
            
            original_cond = original_cond_obj[target_key]
        
        if not isinstance(original_cond, torch.Tensor):
             logging.warning(f"TPSO: Conditioning object is {type(original_cond)}, expected Tensor. Skipping.")
             return (unet,)

        device = devices.device
        dtype = devices.dtype_unet
        
        optimized_cond = original_cond.clone().to(device, dtype=torch.float32).detach()
        optimized_cond.requires_grad_(True)
        
        optimizer = optim.Adam([optimized_cond], lr=tpso_lr)
        target_cond = original_cond.clone().to(device, dtype=torch.float32).detach()
        
        batch_size = original_cond.shape[0]
        
        # Optimization Loop
        if tpso_enabled:
            if batch_size > 1:
                logging.info(f"TPSO: Optimizing batch of {batch_size} for diversity...")
                for step in range(tpso_steps):
                    optimizer.zero_grad()
                    l_sem = F.mse_loss(optimized_cond, target_cond)
                    
                    flat = optimized_cond.view(batch_size, -1)
                    flat_norm = F.normalize(flat, p=2, dim=1)
                    similarity_matrix = torch.mm(flat_norm, flat_norm.t())
                    
                    mask = torch.eye(batch_size, device=device).bool()
                    off_diag = similarity_matrix[~mask]
                    l_div = off_diag.mean() if off_diag.numel() > 0 else torch.tensor(0.0).to(device)
                    
                    loss = l_sem + tpso_lambda * l_div
                    loss.backward()
                    optimizer.step()
                    
                logging.debug(f"TPSO: Optimization done. Final Loss: {loss.item():.4f}")
            else:
                logging.info("TPSO: Single batch. Applying perturbation to escape mode.")
                with torch.no_grad():
                     noise = torch.randn_like(original_cond) * (tpso_lr * 10.0) 
                     optimized_cond = original_cond + noise

        final_optimized_cond = optimized_cond.to(dtype).detach()
        
        # Define max_t outside wrapper for logging and threshold calculation
        max_t = 999.0
        threshold = max_t * (1.0 - tpso_r)

        # 2. Define Wrapper for UNet Forward
        def unet_wrapper(apply_model, args):
            t = args["timestep"]
            t_curr = t[0].item() if torch.is_tensor(t) else t
            
            if t_curr > threshold:
                new_args = args.copy()
                c = args["c"]
                new_c = c.copy()
                
                # Injection logic
                if "c_crossattn" in c:
                    current_emb = c["c_crossattn"]
                    
                    if isinstance(current_emb, torch.Tensor):
                        curr_bs = current_emb.shape[0]
                        target_bs = final_optimized_cond.shape[0]
                        
                        new_emb = current_emb.clone()
                        
                        if curr_bs == 2 * target_bs:
                            # Replace second half (Cond)
                            new_emb[target_bs:] = final_optimized_cond
                        elif curr_bs == target_bs:
                            # Replace all
                            new_emb = final_optimized_cond
                        
                        new_c["c_crossattn"] = new_emb
                        new_args["c"] = new_c
                        return apply_model(new_args["input"], new_args["timestep"], **new_args["c"])
                
                # SDXL case
                if "crossattn" in c and isinstance(c["crossattn"], torch.Tensor):
                     current_emb = c["crossattn"]
                     if current_emb.shape[0] == final_optimized_cond.shape[0]:
                         new_c["crossattn"] = final_optimized_cond
                         new_args["c"] = new_c
                         return apply_model(new_args["input"], new_args["timestep"], **new_args["c"])

            return apply_model(args["input"], args["timestep"], **args["c"])

        patched_unet = unet.clone()
        patched_unet.set_model_unet_function_wrapper(unet_wrapper)
        
        logging.info(f"TPSO: Patch applied. Schedule: t > {threshold:.0f} uses optimized embeddings.")
        return (patched_unet,)

NODE_CLASS_MAPPINGS = {
    "TPSONode": TPSONode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TPSONode": "TPSO Guidance",
}

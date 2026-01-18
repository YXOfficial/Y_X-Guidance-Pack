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
                "tpso_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "tpso_lr": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "tpso_lambda": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tpso_r": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tpso_kappa": ("FLOAT", {"default": 0.8, "min": 0.5, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Training-Free Prompt Semantic Space Optimization (TPSO)."

    def patch(self, unet, p, tpso_enabled=False, tpso_steps=20, tpso_lr=0.01, tpso_lambda=1.0, tpso_r=0.4, tpso_kappa=0.8):
        if not tpso_enabled:
            return (unet,)

        logging.info(f"TPSO: Starting REAL optimization for {len(p.prompts)} prompts (Steps: {tpso_steps})...")
        
        # 1. Preparation
        with torch.no_grad():
            original_cond_obj = p.sd_model.get_learned_conditioning(p.prompts)
        
        device = devices.device
        dtype = devices.dtype_unet
        
        # TPSO works on the main prompt embedding tensor
        # We need to handle both SDXL (dict) and SD1.5 (tensor or dict)
        original_cond_tensor = None
        target_key = "c_crossattn" # Default
        
        if isinstance(original_cond_obj, dict):
            for k in ['crossattn', 'c_crossattn', 'cond']:
                if k in original_cond_obj:
                    target_key = k
                    original_cond_tensor = original_cond_obj[k]
                    break
            if original_cond_tensor is None:
                # Fallback to first 3D tensor
                for k, v in original_cond_obj.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 3:
                        target_key = k
                        original_cond_tensor = v
                        break
        else:
            original_cond_tensor = original_cond_obj

        if original_cond_tensor is None:
            logging.warning("TPSO: Failed to extract conditioning tensor.")
            return (unet,)

        # 2. THE OPTIMIZATION LOOP (The Core of TPSO)
        # We optimize optimized_cond to be diverse yet semantically similar (~kappa)
        
        batch_size = original_cond_tensor.shape[0]
        # We work in float32 for optimization stability
        optimized_cond = original_cond_tensor.clone().to(device, dtype=torch.float32).detach()
        optimized_cond.requires_grad_(True)
        
        target_cond = original_cond_tensor.clone().to(device, dtype=torch.float32).detach()
        optimizer = optim.Adam([optimized_cond], lr=tpso_lr)
        
        # Paper targets: cos(v, v') approx kappa
        kappa = tpso_kappa
        sigma = 0.01 # Tolerance band from paper
        
        for step in range(tpso_steps):
            optimizer.zero_grad()
            
            # Flatten for cosine similarity calculation
            # optimized_cond: [B, L, D] -> [B, L*D]
            v_prime = optimized_cond.view(batch_size, -1)
            v = target_cond.view(batch_size, -1)
            
            # Normalized vectors
            v_prime_norm = F.normalize(v_prime, p=2, dim=1)
            v_norm = F.normalize(v, p=2, dim=1)
            
            # Semantic Alignment Loss (Eq 7)
            # We want cosine similarity between original and optimized to be exactly around kappa
            cos_sim = (v_prime_norm * v_norm).sum(dim=1) # [B]
            l_semantic = torch.max(torch.zeros_like(cos_sim), torch.abs(cos_sim - kappa) - sigma).mean()
            
            # Diversity Loss (Eq 8)
            l_div = torch.tensor(0.0, device=device)
            if batch_size > 1:
                # Maximize distance between batch members
                sim_matrix = torch.mm(v_prime_norm, v_prime_norm.t()) # [B, B]
                mask = torch.eye(batch_size, device=device).bool()
                off_diag = sim_matrix[~mask]
                l_div = off_diag.mean() # Minimize similarity
            else:
                # For single image, we already have diversity by pushing away from the original mode (cos_sim -> kappa)
                # But we can add a small penalty to keep the norm stable
                l_div = torch.abs(torch.norm(v_prime, dim=1) - torch.norm(v, dim=1)).mean() * 0.1

            loss = l_semantic + tpso_lambda * l_div
            
            if not torch.isfinite(loss):
                break
                
            loss.backward()
            optimizer.step()

        # Detach and cast back
        final_optimized_cond = optimized_cond.to(dtype).detach()
        final_original_cond = original_cond_tensor.to(dtype).detach()
        
        # Calculate final similarity for logging
        with torch.no_grad():
            final_v_prime = F.normalize(final_optimized_cond.view(batch_size, -1).float(), p=2, dim=1)
            final_v = F.normalize(final_original_cond.view(batch_size, -1).float(), p=2, dim=1)
            actual_sim = (final_v_prime * final_v).sum(dim=1).mean().item()
        
        logging.info(f"TPSO: Optimization finished. Final Cosine Sim: {actual_sim:.4f} (Target: {kappa})")

        # 3. Injection Wrapper
        max_t = 999.0
        threshold = max_t * (1.0 - tpso_r)

        def unet_wrapper(apply_model, args):
            t = args["timestep"]
            t_curr = t[0].item() if torch.is_tensor(t) else t
            
            if t_curr > threshold:
                new_args = args.copy()
                c = args["c"]
                new_c = c.copy()
                
                # Check for the key we found earlier
                # We prioritize 'c_crossattn' as it's the standard internal key for UNet input
                key_to_replace = "c_crossattn" if "c_crossattn" in c else target_key
                
                if key_to_replace in c:
                    current_emb = c[key_to_replace]
                    
                    if isinstance(current_emb, torch.Tensor):
                        # Detect which slice is the Positive Prompt
                        # In WebUI/ReForge, Positive is usually the SECOND half if CFG is active
                        # batch is [Uncond, Cond]
                        curr_bs = current_emb.shape[0]
                        target_bs = final_optimized_cond.shape[0]
                        
                        new_emb = current_emb.clone()
                        
                        if curr_bs == 2 * target_bs:
                            # Standard [Uncond, Cond]
                            new_emb[target_bs:] = final_optimized_cond
                        elif curr_bs == target_bs:
                            # Cond only
                            new_emb = final_optimized_cond
                        else:
                            # Try to match by value if shapes are weird (e.g. prompt editing)
                            # For simplicity, if we don't know, we don't replace or replace all
                            pass
                        
                        new_c[key_to_replace] = new_emb
                        new_args["c"] = new_c
                        return apply_model(new_args["input"], new_args["timestep"], **new_args["c"])
                
            return apply_model(args["input"], args["timestep"], **args["c"])

        patched_unet = unet.clone()
        patched_unet.set_model_unet_function_wrapper(unet_wrapper)
        
        logging.info(f"TPSO: Patch active for steps t > {threshold:.0f}")
        return (patched_unet,)

NODE_CLASS_MAPPINGS = {
    "TPSONode": TPSONode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TPSONode": "TPSO Guidance",
}
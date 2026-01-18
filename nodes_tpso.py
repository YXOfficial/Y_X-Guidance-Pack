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
                "tpso_lr": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.001}),
                "tpso_lambda": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tpso_r": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tpso_kappa": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Training-Free Prompt Semantic Space Optimization (TPSO) - Faithful Implementation."

    def patch(self, unet, p, tpso_enabled=False, tpso_steps=20, tpso_lr=0.1, tpso_lambda=1.0, tpso_r=0.4, tpso_kappa=0.8):
        if not tpso_enabled:
            return (unet,)

        logging.info(f"TPSO: Starting Optimization (Target Kappa: {tpso_kappa}, Steps: {tpso_steps}, LR: {tpso_lr})...")
        
        with torch.no_grad():
            original_cond_obj = p.sd_model.get_learned_conditioning(p.prompts)
        
        device = devices.device
        dtype = devices.dtype_unet
        
        original_cond_tensor = None
        
        # Extract Tensor Logic
        if isinstance(original_cond_obj, dict):
            for k in ['crossattn', 'c_crossattn', 'cond']:
                if k in original_cond_obj:
                    original_cond_tensor = original_cond_obj[k]
                    break
            if original_cond_tensor is None:
                target_key = next((k for k, v in original_cond_obj.items() if isinstance(v, torch.Tensor) and v.ndim == 3), None)
                if target_key: original_cond_tensor = original_cond_obj[target_key]
        else:
            original_cond_tensor = original_cond_obj

        if original_cond_tensor is None or not isinstance(original_cond_tensor, torch.Tensor):
            return (unet,)

        batch_size = original_cond_tensor.shape[0]
        
        # --- OPTIMIZATION BLOCK ---
        with torch.inference_mode(False):
            with torch.enable_grad():
                target_cond_fp32 = original_cond_tensor.to(device, dtype=torch.float32).detach()
                epsilon = torch.randn_like(target_cond_fp32) * 1e-4
                epsilon.requires_grad_(True)
                optimizer = optim.Adam([epsilon], lr=tpso_lr)
                
                kappa = tpso_kappa
                sigma = 0.01
                
                for step in range(tpso_steps):
                    optimizer.zero_grad()
                    optimized_cond = target_cond_fp32 + epsilon
                    
                    v_prime = optimized_cond.view(batch_size, -1)
                    v = target_cond_fp32.view(batch_size, -1)
                    
                    v_prime_norm = F.normalize(v_prime, p=2, dim=1)
                    v_norm = F.normalize(v, p=2, dim=1)
                    
                    cos_sim = (v_prime_norm * v_norm).sum(dim=1)
                    
                    diff = torch.abs(cos_sim - kappa) - sigma
                    l_semantic = torch.clamp(diff, min=0.0).sum()
                    
                    l_div = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if batch_size > 1:
                        sim_matrix = torch.mm(v_prime_norm, v_prime_norm.t())
                        mask = torch.eye(batch_size, device=device).bool()
                        off_diag = sim_matrix[~mask]
                        if off_diag.numel() > 0:
                            l_div = off_diag.mean()
                    
                    loss = l_semantic + tpso_lambda * l_div
                    loss.backward()
                    optimizer.step()
                    
                    if step == 0 or step == tpso_steps - 1:
                         logging.debug(f"TPSO Step {step}: Loss={loss.item():.4f}, CosSim={cos_sim.mean().item():.4f}")

                final_optimized_cond = (target_cond_fp32 + epsilon).to(dtype).detach()

        # Log final result
        with torch.no_grad():
            final_v_prime = F.normalize(final_optimized_cond.view(batch_size, -1).float(), p=2, dim=1)
            final_v = F.normalize(original_cond_tensor.view(batch_size, -1).float(), p=2, dim=1)
            actual_sim = (final_v_prime * final_v).sum(dim=1).mean().item()
        
        logging.info(f"TPSO: Optimization Finished. Final Cosine Sim: {actual_sim:.4f} (Target: {kappa})")

        # --- WRAPPER WITH DEBUGGING & FORCE INJECTION ---
        max_t = 999.0
        threshold = max_t * (1.0 - tpso_r)
        
        # We define a unique ID for this patch to track it in logs without spamming too much
        patch_id = id(final_optimized_cond)

        def unet_wrapper(apply_model, args):
            t = args["timestep"]
            t_curr = t[0].item() if torch.is_tensor(t) else t
            
            # Debug log only once per threshold crossing to avoid spam
            should_inject = t_curr > threshold
            
            if should_inject:
                c = args["c"]
                injected = False
                
                # Check keys
                for key in ["c_crossattn", "crossattn"]:
                    if key in c and isinstance(c[key], torch.Tensor):
                        current_emb = c[key]
                        
                        # Only proceed if shapes are compatible
                        # final_optimized_cond shape: [Batch_Prompts, Seq_Len, Dim]
                        if current_emb.shape[-1] == final_optimized_cond.shape[-1]:
                            new_c = c.copy()
                            new_emb = current_emb.clone()
                            
                            target_bs = final_optimized_cond.shape[0] # Number of optimized prompts (usually 1)
                            current_bs = current_emb.shape[0]       # Usually 2 (Neg + Pos)
                            
                            # STRATEGY 1: Standard [Uncond, Cond] assumption
                            # If current batch is exactly 2x optimized batch, assume second half is Positive
                            if current_bs == 2 * target_bs:
                                new_emb[target_bs:] = final_optimized_cond
                                injected = True
                                # logging.debug(f"TPSO: Injected via Strategy 1 (Standard Batch Split) at t={t_curr:.0f}")
                            
                            # STRATEGY 2: Exact Match
                            elif current_bs == target_bs:
                                new_emb[:] = final_optimized_cond
                                injected = True
                                # logging.debug(f"TPSO: Injected via Strategy 2 (Exact Replacement) at t={t_curr:.0f}")
                                
                            else:
                                # Fallback: Try to match simply by value? No, let's stick to safe strategies first.
                                pass

                            if injected:
                                new_c[key] = new_emb
                                # IMPORTANT: If we injected into one key, we should check if we need to return immediately
                                # usually c_crossattn is the one.
                                return apply_model(args["input"], args["timestep"], **new_c)
                
                if not injected:
                    # If we expected to inject but didn't find a place, Log it once!
                    # We can use a simple trick to log only occasionally or use the step
                    if int(t_curr) % 100 == 0:
                         logging.warning(f"TPSO: Active (t={t_curr:.0f}) but FAILED to inject! Keys: {list(c.keys())}")

            return apply_model(args["input"], args["timestep"], **args["c"])

        patched_unet = unet.clone()
        patched_unet.set_model_unet_function_wrapper(unet_wrapper)
        return (patched_unet,)

NODE_CLASS_MAPPINGS = {"TPSONode": TPSONode}
NODE_DISPLAY_NAME_MAPPINGS = {"TPSONode": "TPSO Guidance"}
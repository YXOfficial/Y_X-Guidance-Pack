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

        logging.info(f"TPSO: Starting Optimization (Target Kappa: {tpso_kappa}, Steps: {tpso_steps})...")
        
        # Check for inference mode which blocks gradients
        if torch.is_inference_mode_enabled():
            logging.warning("TPSO: Inference Mode detected! Gradients cannot be computed. Optimization skipped.")
            return (unet,)

        with torch.no_grad():
            original_cond_obj = p.sd_model.get_learned_conditioning(p.prompts)
        
        device = devices.device
        dtype = devices.dtype_unet
        
        original_cond_tensor = None
        target_key = "c_crossattn"
        
        if isinstance(original_cond_obj, dict):
            for k in ['crossattn', 'c_crossattn', 'cond']:
                if k in original_cond_obj:
                    target_key = k
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
        
        # --- FAITHFUL OPTIMIZATION (Paper Eq 7 & 8) ---
        # Note: We optimize in Prompt Space as a proxy for Token Space due to architecture constraints
        
        with torch.enable_grad():
            # Init with noise epsilon ~ N(0, 10^-4)
            # We use float32 for precision during optimization
            epsilon = torch.randn_like(original_cond_tensor, device=device, dtype=torch.float32) * 1e-4
            
            # We optimize the *resulting embedding* directly to ensure movement
            # (equivalent to optimizing epsilon in this simplified proxy)
            optimized_cond = (original_cond_tensor.to(device, dtype=torch.float32) + epsilon).detach()
            optimized_cond.requires_grad_(True)
            
            target_cond_fp32 = original_cond_tensor.to(device, dtype=torch.float32).detach()
            
            # Constants
            kappa = tpso_kappa
            sigma = 0.01
            
            # Using manual Gradient Descent for transparency and reliability
            for step in range(tpso_steps):
                v_prime = optimized_cond.view(batch_size, -1)
                v = target_cond_fp32.view(batch_size, -1)
                
                v_prime_norm = F.normalize(v_prime, p=2, dim=1)
                v_norm = F.normalize(v, p=2, dim=1)
                
                # Cosine Similarity
                cos_sim = (v_prime_norm * v_norm).sum(dim=1)
                
                # Semantic Alignment Loss (Eq 7): Sum(max(0, |cos - kappa| - sigma))
                # Using Sum as per paper text
                diff = torch.abs(cos_sim - kappa) - sigma
                l_semantic = torch.clamp(diff, min=0.0).sum()
                
                # Diversity Loss (Eq 8)
                l_div = torch.tensor(0.0, device=device, dtype=torch.float32)
                if batch_size > 1:
                    sim_matrix = torch.mm(v_prime_norm, v_prime_norm.t())
                    # Mask diagonal
                    mask = torch.eye(batch_size, device=device).bool()
                    off_diag = sim_matrix[~mask]
                    if off_diag.numel() > 0:
                        l_div = off_diag.mean() # Paper says sum/N(N-1) which is mean of off-diagonals
                
                loss = l_semantic + tpso_lambda * l_div
                
                if loss.item() < 1e-6:
                    # Converged
                    break
                
                # Compute Gradients manually
                grads = torch.autograd.grad(loss, optimized_cond, create_graph=False)[0]
                
                # Update (Gradient Descent)
                # We want to minimize Loss, so sub_(lr * grad)
                with torch.no_grad():
                    optimized_cond.sub_(grads * tpso_lr)
                
                if step == 0 or step == tpso_steps - 1:
                    logging.debug(f"TPSO Step {step}: Loss={loss.item():.4f}, CosSim={cos_sim.mean().item():.4f}")
            
            final_optimized_cond = optimized_cond.to(dtype).detach()

        # Log final stats
        with torch.no_grad():
            final_v_prime = F.normalize(final_optimized_cond.view(batch_size, -1).float(), p=2, dim=1)
            final_v = F.normalize(original_cond_tensor.view(batch_size, -1).float(), p=2, dim=1)
            actual_sim = (final_v_prime * final_v).sum(dim=1).mean().item()
        
        logging.info(f"TPSO: Optimization Finished. Final Cosine Sim: {actual_sim:.4f} (Target: {kappa})")

        # --- WRAPPER & INJECTION ---
        max_t = 999.0
        threshold = max_t * (1.0 - tpso_r)
        ref_cond = original_cond_tensor.to(device, dtype=dtype).detach()

        def unet_wrapper(apply_model, args):
            t = args["timestep"]
            t_curr = t[0].item() if torch.is_tensor(t) else t
            
            if t_curr > threshold:
                c = args["c"]
                for key in ["c_crossattn", "crossattn"]:
                    if key in c and isinstance(c[key], torch.Tensor):
                        current_emb = c[key]
                        if current_emb.shape[-1] == final_optimized_cond.shape[-1]:
                            new_c = c.copy()
                            new_emb = current_emb.clone()
                            
                            target_bs = final_optimized_cond.shape[0]
                            # Value-based matching to inject optimized prompts
                            for i in range(current_emb.shape[0]):
                                slice_to_check = current_emb[i:i+1]
                                for j in range(target_bs):
                                    # Relaxed tolerance slightly for float16 matching
                                    if torch.allclose(slice_to_check.float(), ref_cond[j:j+1].float(), atol=1e-3):
                                        new_emb[i] = final_optimized_cond[j]
                                        break
                            
                            new_c[key] = new_emb
                            return apply_model(args["input"], args["timestep"], **new_c)
                
            return apply_model(args["input"], args["timestep"], **args["c"])

        patched_unet = unet.clone()
        patched_unet.set_model_unet_function_wrapper(unet_wrapper)
        return (patched_unet,)

NODE_CLASS_MAPPINGS = {"TPSONode": TPSONode}
NODE_DISPLAY_NAME_MAPPINGS = {"TPSONode": "TPSO Guidance"}
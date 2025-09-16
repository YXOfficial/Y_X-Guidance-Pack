
# --- Guidance-Pack: simplified unified guidance (Mahiro + basic CFG-Zero hook) ---
import logging
import torch
import torch.nn.functional as F

class CFGZeroNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            # toggles
            "mahiro_enabled": ("BOOLEAN", {"default": False}),
            "cfg_zero_enabled": ("BOOLEAN", {"default": False}),
            "zero_init_first_step": ("BOOLEAN", {"default": False}),
            # placeholders to keep UI compatible (no-ops in this simplified build)
            "fdg_enabled": ("BOOLEAN", {"default": False}),
            "w_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "w_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "fdg_levels": ("INT", {"default": 3, "min": 1, "max": 6}),
            "s2_guidance_enabled": ("BOOLEAN", {"default": False}),
            "s2_scale_omega": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
            "s2_drop_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Unified guidance hook for Test-reForge WebUI: basic CFG-Zero (projective scale) + optional Mahiro gating. FDG/S² placeholders to keep UI stable."

    def patch(self, model,
              mahiro_enabled: bool = False,
              cfg_zero_enabled: bool = False,
              zero_init_first_step: bool = False,
              fdg_enabled: bool = False, w_low: float = 1.0, w_high: float = 1.0, fdg_levels: int = 3,
              s2_guidance_enabled: bool = False, s2_scale_omega: float = 0.25, s2_drop_ratio: float = 0.1):

        m = model.clone()

        try:
            initial_sigma = m.model.model_sampling.sigma_max
        except Exception:
            initial_sigma = None

        def guidance_function(args):
            # expected keys in Test‑reForge post‑cfg
            C: torch.Tensor = args['cond_denoised']
            U: torch.Tensor = args['uncond_denoised']
            scale: float = float(args.get('cond_scale', 1.0))

            # optional zero-init for the very first step
            if cfg_zero_enabled and zero_init_first_step and initial_sigma is not None:
                sigma = args.get('sigma', None)
                if isinstance(sigma, torch.Tensor) and sigma.numel() > 0:
                    if abs(float(sigma.flatten()[0].item()) - float(initial_sigma)) < 1e-5:
                        return U

            # --- Base: classic CFG
            base = U + (C - U) * scale

            # --- Basic "CFG-Zero" style projection (lightweight):
            # project conditional onto uncond direction to estimate per-batch scalar, then re-scale
            if cfg_zero_enabled:
                b = C.reshape(C.shape[0], -1)
                a = U.reshape(U.shape[0], -1)
                # st* = <b,a> / ||a||^2
                dot = (b * a).sum(dim=1, keepdim=True)
                denom = (a * a).sum(dim=1, keepdim=True) + 1e-8
                st = (dot / denom).clamp(min=0.0, max=4.0)  # safe bounds
                # effective scale per-batch
                eff_scale = scale * st.view(-1, 1, 1, 1)
                base = U + (C - U) * eff_scale

            combined = base

            # --- Mahiro gating (optional): blend between pure-cond leap and current guidance by cosine‑sim
            if mahiro_enabled:
                leap = C * scale
                merge = 0.5 * (leap + combined)

                def srs(x: torch.Tensor):
                    return torch.sqrt(x.abs() + 1e-12) * x.sign()

                u_leap = U * scale
                # cosine over channel/spatial dimensions; mean over batch
                dim = list(range(1, u_leap.ndim))
                sim = F.cosine_similarity(srs(u_leap), srs(merge), dim=dim).mean()
                gate = 2.0 * (sim + 1.0)  # [0,4]
                combined = (gate * combined + (4.0 - gate) * leap) / 4.0

            # FDG/S² are no‑ops in this simplified build to ensure stability
            return combined

        m.set_model_sampler_post_cfg_function(guidance_function, "guidance_pack_unified")
        return (m,)

NODE_CLASS_MAPPINGS = {"CFGZeroFDGS2Node": CFGZeroNode}
NODE_DISPLAY_NAME_MAPPINGS = {"CFGZeroFDGS2Node": "Guidance Pack (Mahiro / basic CFG-Zero)"} 

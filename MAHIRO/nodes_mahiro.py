
# --- START OF FILE nodes_mahiro.py ---
import torch
import torch.nn.functional as F

class YX_MahiroGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Mahiro guidance: blend CFG and pure-cond (leap) by cosine similarity of signed-sqrt normalized directions."

    def patch(self, model, enabled: bool = True):
        if not enabled:
            return (model,)

        m = model.clone()

        def mahiro_post_cfg(args):
            # available keys: input, denoised, cond_denoised, uncond_denoised, cond_scale, sigma, cond (list of conds)
            scale: float = float(args.get("cond_scale", 1.0))
            C: torch.Tensor = args["cond_denoised"]
            U: torch.Tensor = args["uncond_denoised"]
            X: torch.Tensor = args["input"]

            # pure conditional leap (ignores U) and classic CFG
            leap = C * scale
            cfg  = U + (C - U) * scale

            # average two "hints"
            merge = 0.5 * (leap + cfg)

            # signed sqrt normalization (reduce magnitude bias, focus on direction)
            def srs(x: torch.Tensor):
                return torch.sqrt(x.abs() + 1e-12) * x.sign()

            # similarity between uncond leap and merged hint
            u_leap = U * scale
            sim = F.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, U.ndim))).mean()

            # map sim∈[-1,1] → [0,4], then gate between CFG and LEAP
            simsc = 2.0 * (sim + 1.0)  # [0,4]
            wm = (simsc * cfg + (4.0 - simsc) * leap) / 4.0

            # return "denoised" such that to_d(x, sigma, den) = wm / sigma
            # i.e. den = X - wm
            return X - wm

        # install as post-CFG function so it can see C, U, scale
        m.set_model_sampler_post_cfg_function(mahiro_post_cfg, "yx_mahiro_guidance")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "YX_MahiroGuidance": YX_MahiroGuidance
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YX_MahiroGuidance": "YX-Mahiro Guidance"
}
# --- END OF FILE nodes_mahiro.py ---

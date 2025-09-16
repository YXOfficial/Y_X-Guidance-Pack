
import torch
import torch.nn.functional as F

class AppendMahiroPostCFG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "enabled": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"

    def patch(self, model, enabled: bool=True):
        if not enabled:
            return (model,)
        # Do NOT clone again; append post-cfg to the same patched UNet so both run sequentially
        m = model

        def mahiro_post(args):
            scale: float = float(args.get("cond_scale", 1.0))
            C: torch.Tensor = args["cond_denoised"]
            U: torch.Tensor = args["uncond_denoised"]
            cfg: torch.Tensor = args["denoised"]  # current CFG result

            leap = C * scale
            merge = 0.5 * (leap + cfg)

            def srs(x: torch.Tensor):
                return torch.sqrt(x.abs() + 1e-12) * x.sign()

            u_leap = U * scale
            sim = F.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, u_leap.ndim))).mean()
            gate = 2.0 * (sim + 1.0)
            return (gate * cfg + (4.0 - gate) * leap) / 4.0

        m.set_model_sampler_post_cfg_function(mahiro_post)
        return (m,)

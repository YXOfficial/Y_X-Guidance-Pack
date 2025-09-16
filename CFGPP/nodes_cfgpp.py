
# --- START OF FILE nodes_cfgpp.py ---
import torch

class YX_CFGPlusHook:
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
    DESCRIPTION = "Enable CFG++ compatibility hook (exposes uncond_denoised via post-cfg callback for downstream samplers)."

    def patch(self, model, enabled: bool=True):
        if not enabled:
            return (model,)

        m = model.clone()

        # This hook just exposes uncond_denoised to the sampling loop; it does not change the guidance math by itself.
        # Reference samplers (Euler/DPM++ CFG++) included in this pack import their own post-cfg capture.
        temp = {"uncond": None}

        def post_cfg_capture(args):
            temp["uncond"] = args["uncond_denoised"]
            return args["denoised"]

        m.set_model_sampler_post_cfg_function(post_cfg_capture, "yx_cfgpp_capture")

        return (m,)

NODE_CLASS_MAPPINGS = {"YX_CFGPlusHook": YX_CFGPlusHook}
NODE_DISPLAY_NAME_MAPPINGS = {"YX_CFGPlusHook": "YX-CFG++ Hook (compat)"}
# --- END OF FILE nodes_cfgpp.py ---

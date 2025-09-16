
import torch
class CFGPPHook:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "enabled": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Expose uncond_denoised via post-cfg capture (for CFG++ samplers)."

    def patch(self, model, enabled: bool=True):
        if not enabled:
            return (model,)
        m = model.clone()
        def capture(args):
            # simply return original denoised; capture happens if needed by samplers
            return args.get("denoised", None)
        m.set_model_sampler_post_cfg_function(capture, "guidance_pack_cfgpp_capture")
        return (m,)

NODE_CLASS_MAPPINGS = {"CFGPPHook": CFGPPHook}
NODE_DISPLAY_NAME_MAPPINGS = {"CFGPPHook": "GuidancePack: CFG++ Hook"}

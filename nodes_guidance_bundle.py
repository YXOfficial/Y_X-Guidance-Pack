import logging

from guidance_utils import (
    ensure_guidance_pipeline,
    get_initial_sigma,
    make_cfg_zero_base_builder,
    make_fdg_modifier,
    make_s2_modifier,
    make_zeresfdg_base_builder,
    make_zeresfdg_modifier,
)


class GuidanceBundleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_zero_enabled": ("BOOLEAN", {"default": False}),
                "zero_init_first_step": ("BOOLEAN", {"default": False}),
                "fdg_enabled": ("BOOLEAN", {"default": False}),
                "fdg_w_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "fdg_w_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "fdg_levels": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
                "s2_guidance_enabled": ("BOOLEAN", {"default": False}),
                "s2_scale_omega": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.01}),
                "s2_drop_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "round": 0.01}),
                "zeresfdg_enabled": ("BOOLEAN", {"default": False}),
                "zeresfdg_w_low": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "zeresfdg_w_high": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "zeresfdg_alpha": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "zeresfdg_tau_lo": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "zeresfdg_tau_hi": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "zeresfdg_beta": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "zeresfdg_controller_enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Bundles CFG-Zero, FDG, S², and ZeResFDG into a single configurable node."

    def patch(
        self,
        model,
        cfg_zero_enabled: bool = False,
        zero_init_first_step: bool = False,
        fdg_enabled: bool = False,
        fdg_w_low: float = 1.0,
        fdg_w_high: float = 1.0,
        fdg_levels: int = 3,
        s2_guidance_enabled: bool = False,
        s2_scale_omega: float = 0.25,
        s2_drop_ratio: float = 0.1,
        zeresfdg_enabled: bool = False,
        zeresfdg_w_low: float = 0.6,
        zeresfdg_w_high: float = 1.3,
        zeresfdg_alpha: float = 0.7,
        zeresfdg_tau_lo: float = 0.45,
        zeresfdg_tau_hi: float = 0.6,
        zeresfdg_beta: float = 0.8,
        zeresfdg_controller_enabled: bool = True,
    ):
        if not any(
            [
                cfg_zero_enabled,
                fdg_enabled,
                s2_guidance_enabled,
                zeresfdg_enabled,
            ]
        ):
            return (model,)

        patched_model = model.clone()

        if hasattr(model, "_guidance_pipeline"):
            patched_model._guidance_pipeline = model._guidance_pipeline
            patched_model.set_model_sampler_post_cfg_function(
                patched_model._guidance_pipeline.run, "custom_guidance_pipeline"
            )

        pipeline = ensure_guidance_pipeline(patched_model)

        if zeresfdg_enabled:
            pipeline.set_base_builder(make_zeresfdg_base_builder())
        elif cfg_zero_enabled:
            initial_sigma = get_initial_sigma(patched_model)
            pipeline.set_base_builder(
                make_cfg_zero_base_builder(zero_init_first_step, initial_sigma)
            )

        if fdg_enabled:
            pipeline.add_modifier("fdg", make_fdg_modifier(fdg_w_low, fdg_w_high, int(fdg_levels)))

        if s2_guidance_enabled:
            pipeline.add_modifier(
                "s2_guidance",
                make_s2_modifier(patched_model, s2_drop_ratio, s2_scale_omega),
            )

        if zeresfdg_enabled:
            pipeline.add_modifier(
                "zeresfdg",
                make_zeresfdg_modifier(
                    w_low=zeresfdg_w_low,
                    w_high=zeresfdg_w_high,
                    alpha=zeresfdg_alpha,
                    tau_lo=zeresfdg_tau_lo,
                    tau_hi=zeresfdg_tau_hi,
                    beta=zeresfdg_beta,
                    controller_enabled=zeresfdg_controller_enabled,
                ),
            )

        logging.debug(
            "Patched model with bundle cfg_zero=%s, fdg=%s, s2_guidance=%s, zeresfdg=%s",
            cfg_zero_enabled,
            fdg_enabled,
            s2_guidance_enabled,
            zeresfdg_enabled,
        )

        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "GuidanceBundleNode": GuidanceBundleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GuidanceBundleNode": "YX-Guidance Bundle",
}

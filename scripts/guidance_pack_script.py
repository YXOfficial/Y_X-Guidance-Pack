import logging
import sys
import traceback
from functools import partial

import gradio as gr
from modules import scripts, script_callbacks

from yx_guidance_utils import (
    clear_generation_params_once,
    ensure_guidance_pipeline,
    make_qsilk_modifier,
    reset_unet_if_needed,
)
from nodes_cfg_zero import CFGZeroNode
from nodes_fdg import FDGNode
from nodes_zeresfdg import ZeResFDGNode
from nodes_s2_guidance import S2GuidanceNode
from nodes_tpso import TPSONode


class GuidancePackScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.cfg_zero_enabled = False
        self.zero_init_first_step = False

        self.fdg_enabled = False
        self.fdg_w_low = 1.0
        self.fdg_w_high = 1.0
        self.fdg_levels = 3

        self.zeresfdg_enabled = False
        self.zeresfdg_controller_enabled = True
        self.zeresfdg_w_low = 0.6
        self.zeresfdg_w_high = 1.3
        self.zeresfdg_alpha = 0.7
        self.zeresfdg_tau_lo = 0.45
        self.zeresfdg_tau_hi = 0.6
        self.zeresfdg_beta = 0.8

        self.s2_guidance_enabled = False
        self.s2_scale_omega = 0.25
        self.s2_drop_ratio = 0.1

        self.qsilk_enabled = False
        self.micro_q_low = 0.001
        self.micro_q_high = 0.999
        self.micro_alpha = 2.0
        self.qsilk_vp_mix = 0.15
        self.qsilk_per_channel = False
        self.qsilk_late_weighting = True
        self.qsilk_use_aqclip = False
        self.qsilk_tile_size = 32
        self.qsilk_stride = 16
        self.qsilk_aqclip_alpha = 2.0
        self.qsilk_ema_beta = 0.8

        self.tpso_enabled = False
        self.tpso_steps = 10
        self.tpso_lr = 0.01
        self.tpso_lambda = 0.5
        self.tpso_r = 0.4
        self.tpso_kappa = 0.8

        self.cfg_zero_node = CFGZeroNode()
        self.fdg_node = FDGNode()
        self.zeresfdg_node = ZeResFDGNode()
        self.s2_guidance_node = S2GuidanceNode()
        self.tpso_node = TPSONode()

    sorting_priority = 15.0

    def title(self):
        return "Guidance Pack"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Unified UI for CFG-Zero, FDG, ZeResFDG, S²-Guidance, and TPSO.")
            with gr.Tabs():
                with gr.Tab(label="CFG-Zero"):
                    gr.Markdown("Enable CFG-Zero guidance adjustments.")
                    cfg_zero_enabled = gr.Checkbox(label="Enable CFG-Zero", value=self.cfg_zero_enabled)
                    zero_init_first_step = gr.Checkbox(
                        label="Zero Init First Step (Experimental)", value=self.zero_init_first_step
                    )

                with gr.Tab(label="FDG"):
                    gr.Markdown("Configure Frequency-Decoupled Guidance.")
                    fdg_enabled = gr.Checkbox(label="Enable FDG", value=self.fdg_enabled)
                    fdg_w_low = gr.Slider(
                        label="w_low (Low-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.fdg_w_low
                    )
                    fdg_w_high = gr.Slider(
                        label="w_high (High-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.fdg_w_high
                    )
                    fdg_levels = gr.Slider(
                        label="FDG Pyramid Levels", minimum=2, maximum=8, step=1, value=self.fdg_levels
                    )

                with gr.Tab(label="ZeResFDG"):
                    gr.Markdown(
                        "Enable ZeResFDG guidance with spectral EMA switching between CFGZeroFD and RescaleFDG modes."
                    )
                    zeresfdg_enabled = gr.Checkbox(label="Enable ZeResFDG", value=self.zeresfdg_enabled)
                    zeresfdg_controller_enabled = gr.Checkbox(
                        label="Enable Spectral Controller", value=self.zeresfdg_controller_enabled
                    )
                    zeresfdg_w_low = gr.Slider(
                        label="λ_l (Low-Frequency Gain)", minimum=0.0, maximum=10.0, step=0.05, value=self.zeresfdg_w_low
                    )
                    zeresfdg_w_high = gr.Slider(
                        label="λ_h (High-Frequency Gain)", minimum=0.0, maximum=10.0, step=0.05, value=self.zeresfdg_w_high
                    )
                    zeresfdg_alpha = gr.Slider(
                        label="α (Rescale Blend)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_alpha
                    )
                    zeresfdg_tau_lo = gr.Slider(
                        label="τ_lo (EMA Low Threshold)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_tau_lo
                    )
                    zeresfdg_tau_hi = gr.Slider(
                        label="τ_hi (EMA High Threshold)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_tau_hi
                    )
                    zeresfdg_beta = gr.Slider(
                        label="β (EMA Smoothing)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_beta
                    )

                with gr.Tab(label="S²-Guidance"):
                    gr.Markdown("Enable and configure S²-Guidance. Increases generation time.")
                    s2_guidance_enabled = gr.Checkbox(
                        label="Enable S²-Guidance", value=self.s2_guidance_enabled
                    )
                    s2_scale_omega = gr.Slider(
                        label="S² Scale (ω)", minimum=0.0, maximum=2.0, step=0.05, value=self.s2_scale_omega
                    )
                    s2_drop_ratio = gr.Slider(
                        label="Block Drop Ratio",
                        minimum=0.0,
                        maximum=0.5,
                        step=0.01,
                        value=self.s2_drop_ratio,
                        info="Ratio of UNet/DiT blocks to drop. Paper suggests ~0.1 (10%)",
                    )

                with gr.Tab(label="QSilk"):
                    gr.Markdown("Stabilize denoised latents with Micrograin + AQClip-Lite.")
                    qsilk_enabled = gr.Checkbox(label="Enable QSilk", value=self.qsilk_enabled)
                    micro_q_low = gr.Slider(
                        label="Micrograin q_low",
                        minimum=0.0,
                        maximum=0.01,
                        step=0.0001,
                        value=self.micro_q_low,
                    )
                    micro_q_high = gr.Slider(
                        label="Micrograin q_high",
                        minimum=0.95,
                        maximum=1.0,
                        step=0.0001,
                        value=self.micro_q_high,
                    )
                    micro_alpha = gr.Slider(
                        label="Micrograin α",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=self.micro_alpha,
                    )
                    qsilk_vp_mix = gr.Slider(
                        label="Micrograin VP-Mix",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=self.qsilk_vp_mix,
                        info="Blend toward variance-preserving remap (0=off, default 0.15).",
                    )
                    qsilk_per_channel = gr.Checkbox(
                        label="Micrograin Per-Channel (slower)",
                        value=self.qsilk_per_channel,
                        info="Compute quantiles/VP stats per channel; can be slower.",
                    )
                    qsilk_late_weighting = gr.Checkbox(
                        label="Late-Step Weighting",
                        value=self.qsilk_late_weighting,
                        info="Fade in micrograin over sampling steps to reduce early color feedback.",
                    )
                    qsilk_use_aqclip = gr.Checkbox(
                        label="Enable AQClip-Lite", value=self.qsilk_use_aqclip
                    )
                    qsilk_tile_size = gr.Slider(
                        label="AQClip Tile Size",
                        minimum=8,
                        maximum=128,
                        step=8,
                        value=self.qsilk_tile_size,
                    )
                    qsilk_stride = gr.Slider(
                        label="AQClip Stride",
                        minimum=4,
                        maximum=128,
                        step=4,
                        value=self.qsilk_stride,
                        info="Should be ≤ tile size to allow overlap",
                    )
                    qsilk_aqclip_alpha = gr.Slider(
                        label="AQClip α",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=self.qsilk_aqclip_alpha,
                    )
                    qsilk_ema_beta = gr.Slider(
                        label="AQClip EMA β",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=self.qsilk_ema_beta,
                    )
                
                with gr.Tab(label="TPSO"):
                    gr.Markdown("Token-Prompt embedding Space Optimization (Diversity).")
                    tpso_enabled = gr.Checkbox(label="Enable TPSO", value=self.tpso_enabled)
                    tpso_steps = gr.Slider(
                        label="Optimization Steps", minimum=1, maximum=50, step=1, value=self.tpso_steps
                    )
                    tpso_lr = gr.Slider(
                        label="Learning Rate", minimum=0.001, maximum=0.1, step=0.001, value=self.tpso_lr
                    )
                    tpso_lambda = gr.Slider(
                        label="Diversity Lambda", minimum=0.0, maximum=5.0, step=0.1, value=self.tpso_lambda
                    )
                    tpso_r = gr.Slider(
                        label="Schedule Ratio (r)", minimum=0.0, maximum=1.0, step=0.05, value=self.tpso_r,
                        info="Proportion of early steps to use optimized embeddings (0.4 = 40%)."
                    )

        cfg_zero_enabled.change(lambda x: setattr(self, "cfg_zero_enabled", x), inputs=[cfg_zero_enabled], outputs=None)
        zero_init_first_step.change(
            lambda x: setattr(self, "zero_init_first_step", x), inputs=[zero_init_first_step], outputs=None
        )

        fdg_enabled.change(lambda x: setattr(self, "fdg_enabled", x), inputs=[fdg_enabled], outputs=None)
        fdg_w_low.change(lambda x: setattr(self, "fdg_w_low", x), inputs=[fdg_w_low], outputs=None)
        fdg_w_high.change(lambda x: setattr(self, "fdg_w_high", x), inputs=[fdg_w_high], outputs=None)
        fdg_levels.change(lambda x: setattr(self, "fdg_levels", x), inputs=[fdg_levels], outputs=None)

        zeresfdg_enabled.change(lambda x: setattr(self, "zeresfdg_enabled", x), inputs=[zeresfdg_enabled], outputs=None)
        zeresfdg_controller_enabled.change(
            lambda x: setattr(self, "zeresfdg_controller_enabled", x), inputs=[zeresfdg_controller_enabled], outputs=None
        )
        zeresfdg_w_low.change(lambda x: setattr(self, "zeresfdg_w_low", x), inputs=[zeresfdg_w_low], outputs=None)
        zeresfdg_w_high.change(lambda x: setattr(self, "zeresfdg_w_high", x), inputs=[zeresfdg_w_high], outputs=None)
        zeresfdg_alpha.change(lambda x: setattr(self, "zeresfdg_alpha", x), inputs=[zeresfdg_alpha], outputs=None)
        zeresfdg_tau_lo.change(lambda x: setattr(self, "zeresfdg_tau_lo", x), inputs=[zeresfdg_tau_lo], outputs=None)
        zeresfdg_tau_hi.change(lambda x: setattr(self, "zeresfdg_tau_hi", x), inputs=[zeresfdg_tau_hi], outputs=None)
        zeresfdg_beta.change(lambda x: setattr(self, "zeresfdg_beta", x), inputs=[zeresfdg_beta], outputs=None)

        s2_guidance_enabled.change(
            lambda x: setattr(self, "s2_guidance_enabled", x), inputs=[s2_guidance_enabled], outputs=None
        )
        s2_scale_omega.change(lambda x: setattr(self, "s2_scale_omega", x), inputs=[s2_scale_omega], outputs=None)
        s2_drop_ratio.change(lambda x: setattr(self, "s2_drop_ratio", x), inputs=[s2_drop_ratio], outputs=None)

        qsilk_enabled.change(lambda x: setattr(self, "qsilk_enabled", x), inputs=[qsilk_enabled], outputs=None)
        micro_q_low.change(lambda x: setattr(self, "micro_q_low", x), inputs=[micro_q_low], outputs=None)
        micro_q_high.change(lambda x: setattr(self, "micro_q_high", x), inputs=[micro_q_high], outputs=None)
        micro_alpha.change(lambda x: setattr(self, "micro_alpha", x), inputs=[micro_alpha], outputs=None)
        qsilk_vp_mix.change(lambda x: setattr(self, "qsilk_vp_mix", x), inputs=[qsilk_vp_mix], outputs=None)
        qsilk_per_channel.change(
            lambda x: setattr(self, "qsilk_per_channel", x), inputs=[qsilk_per_channel], outputs=None
        )
        qsilk_late_weighting.change(
            lambda x: setattr(self, "qsilk_late_weighting", x), inputs=[qsilk_late_weighting], outputs=None
        )
        qsilk_use_aqclip.change(
            lambda x: setattr(self, "qsilk_use_aqclip", x), inputs=[qsilk_use_aqclip], outputs=None
        )
        qsilk_tile_size.change(
            lambda x: setattr(self, "qsilk_tile_size", x), inputs=[qsilk_tile_size], outputs=None
        )
        qsilk_stride.change(lambda x: setattr(self, "qsilk_stride", x), inputs=[qsilk_stride], outputs=None)
        qsilk_aqclip_alpha.change(
            lambda x: setattr(self, "qsilk_aqclip_alpha", x), inputs=[qsilk_aqclip_alpha], outputs=None
        )
        qsilk_ema_beta.change(lambda x: setattr(self, "qsilk_ema_beta", x), inputs=[qsilk_ema_beta], outputs=None)

        tpso_enabled.change(lambda x: setattr(self, "tpso_enabled", x), inputs=[tpso_enabled], outputs=None)
        tpso_steps.change(lambda x: setattr(self, "tpso_steps", x), inputs=[tpso_steps], outputs=None)
        tpso_lr.change(lambda x: setattr(self, "tpso_lr", x), inputs=[tpso_lr], outputs=None)
        tpso_lambda.change(lambda x: setattr(self, "tpso_lambda", x), inputs=[tpso_lambda], outputs=None)
        tpso_r.change(lambda x: setattr(self, "tpso_r", x), inputs=[tpso_r], outputs=None)

        self.ui_controls = [
            cfg_zero_enabled,
            zero_init_first_step,
            fdg_enabled,
            fdg_w_low,
            fdg_w_high,
            fdg_levels,
            zeresfdg_enabled,
            zeresfdg_controller_enabled,
            zeresfdg_w_low,
            zeresfdg_w_high,
            zeresfdg_alpha,
            zeresfdg_tau_lo,
            zeresfdg_tau_hi,
            zeresfdg_beta,
            s2_guidance_enabled,
            s2_scale_omega,
            s2_drop_ratio,
            qsilk_enabled,
            micro_q_low,
            micro_q_high,
            micro_alpha,
            qsilk_vp_mix,
            qsilk_per_channel,
            qsilk_late_weighting,
            qsilk_use_aqclip,
            qsilk_tile_size,
            qsilk_stride,
            qsilk_aqclip_alpha,
            qsilk_ema_beta,
            tpso_enabled,
            tpso_steps,
            tpso_lr,
            tpso_lambda,
            tpso_r,
        ]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        expected_args = len(self.ui_controls)
        if len(args) >= expected_args:
            (
                self.cfg_zero_enabled,
                self.zero_init_first_step,
                self.fdg_enabled,
                self.fdg_w_low,
                self.fdg_w_high,
                self.fdg_levels,
                self.zeresfdg_enabled,
                self.zeresfdg_controller_enabled,
                self.zeresfdg_w_low,
                self.zeresfdg_w_high,
                self.zeresfdg_alpha,
                self.zeresfdg_tau_lo,
                self.zeresfdg_tau_hi,
                self.zeresfdg_beta,
                self.s2_guidance_enabled,
                self.s2_scale_omega,
                self.s2_drop_ratio,
                self.qsilk_enabled,
                self.micro_q_low,
                self.micro_q_high,
                self.micro_alpha,
                self.qsilk_vp_mix,
                self.qsilk_per_channel,
                self.qsilk_late_weighting,
                self.qsilk_use_aqclip,
                self.qsilk_tile_size,
                self.qsilk_stride,
                self.qsilk_aqclip_alpha,
                self.qsilk_ema_beta,
                self.tpso_enabled,
                self.tpso_steps,
                self.tpso_lr,
                self.tpso_lambda,
                self.tpso_r,
            ) = args[:expected_args]
        else:
            logging.warning("Guidance Pack: Not enough arguments provided from UI.")

        xyz_settings = getattr(p, "_guidance_xyz", {})
        cfg_zero_xyz = xyz_settings.get("cfg_zero", {})
        if "cfg_zero_enabled" in cfg_zero_xyz:
            self.cfg_zero_enabled = str(cfg_zero_xyz["cfg_zero_enabled"]).lower() == "true"
        if "zero_init" in cfg_zero_xyz:
            self.zero_init_first_step = str(cfg_zero_xyz["zero_init"]).lower() == "true"

        fdg_xyz = xyz_settings.get("fdg", {})
        if "fdg_enabled" in fdg_xyz:
            self.fdg_enabled = str(fdg_xyz["fdg_enabled"]).lower() == "true"
        if "w_low" in fdg_xyz:
            self.fdg_w_low = float(fdg_xyz["w_low"])
        if "w_high" in fdg_xyz:
            self.fdg_w_high = float(fdg_xyz["w_high"])
        if "fdg_levels" in fdg_xyz:
            self.fdg_levels = int(fdg_xyz["fdg_levels"])

        zeresfdg_xyz = xyz_settings.get("zeresfdg", {})
        if "zeresfdg_enabled" in zeresfdg_xyz:
            self.zeresfdg_enabled = str(zeresfdg_xyz["zeresfdg_enabled"]).lower() == "true"
        if "controller_enabled" in zeresfdg_xyz:
            self.zeresfdg_controller_enabled = str(zeresfdg_xyz["controller_enabled"]).lower() == "true"
        if "w_low" in zeresfdg_xyz:
            self.zeresfdg_w_low = float(zeresfdg_xyz["w_low"])
        if "w_high" in zeresfdg_xyz:
            self.zeresfdg_w_high = float(zeresfdg_xyz["w_high"])
        if "alpha" in zeresfdg_xyz:
            self.zeresfdg_alpha = float(zeresfdg_xyz["alpha"])
        if "tau_lo" in zeresfdg_xyz:
            self.zeresfdg_tau_lo = float(zeresfdg_xyz["tau_lo"])
        if "tau_hi" in zeresfdg_xyz:
            self.zeresfdg_tau_hi = float(zeresfdg_xyz["tau_hi"])
        if "beta" in zeresfdg_xyz:
            self.zeresfdg_beta = float(zeresfdg_xyz["beta"])

        s2_guidance_xyz = xyz_settings.get("s2_guidance", {})
        if "s2_enabled" in s2_guidance_xyz:
            self.s2_guidance_enabled = str(s2_guidance_xyz["s2_enabled"]).lower() == "true"
        if "s2_omega" in s2_guidance_xyz:
            self.s2_scale_omega = float(s2_guidance_xyz["s2_omega"])
        if "s2_drop" in s2_guidance_xyz:
            self.s2_drop_ratio = float(s2_guidance_xyz["s2_drop"])

        qsilk_xyz = xyz_settings.get("qsilk", {})
        if "qsilk_enabled" in qsilk_xyz:
            self.qsilk_enabled = str(qsilk_xyz["qsilk_enabled"]).lower() == "true"
        if "micro_q_low" in qsilk_xyz:
            self.micro_q_low = float(qsilk_xyz["micro_q_low"])
        if "micro_q_high" in qsilk_xyz:
            self.micro_q_high = float(qsilk_xyz["micro_q_high"])
        if "micro_alpha" in qsilk_xyz:
            self.micro_alpha = float(qsilk_xyz["micro_alpha"])
        if "micro_vp_mix" in qsilk_xyz:
            self.qsilk_vp_mix = float(qsilk_xyz["micro_vp_mix"])
        if "per_channel" in qsilk_xyz:
            self.qsilk_per_channel = str(qsilk_xyz["per_channel"]).lower() == "true"
        if "late_weighting" in qsilk_xyz:
            self.qsilk_late_weighting = str(qsilk_xyz["late_weighting"]).lower() == "true"
        if "use_aqclip" in qsilk_xyz:
            self.qsilk_use_aqclip = str(qsilk_xyz["use_aqclip"]).lower() == "true"
        if "tile_size" in qsilk_xyz:
            self.qsilk_tile_size = int(qsilk_xyz["tile_size"])
        if "stride" in qsilk_xyz:
            self.qsilk_stride = int(qsilk_xyz["stride"])
        if "aqclip_alpha" in qsilk_xyz:
            self.qsilk_aqclip_alpha = float(qsilk_xyz["aqclip_alpha"])
        if "ema_beta" in qsilk_xyz:
            self.qsilk_ema_beta = float(qsilk_xyz["ema_beta"])

        tpso_xyz = xyz_settings.get("tpso", {})
        if "tpso_enabled" in tpso_xyz:
            self.tpso_enabled = str(tpso_xyz["tpso_enabled"]).lower() == "true"
        if "tpso_steps" in tpso_xyz:
            self.tpso_steps = int(tpso_xyz["tpso_steps"])
        if "tpso_lr" in tpso_xyz:
            self.tpso_lr = float(tpso_xyz["tpso_lr"])
        if "tpso_lambda" in tpso_xyz:
            self.tpso_lambda = float(tpso_xyz["tpso_lambda"])
        if "tpso_r" in tpso_xyz:
            self.tpso_r = float(tpso_xyz["tpso_r"])

        restored = reset_unet_if_needed(p)
        if restored:
            clear_generation_params_once(p)

        if self.cfg_zero_enabled:
            patched_unet = self.cfg_zero_node.patch(
                p.sd_model.forge_objects.unet,
                cfg_zero_enabled=self.cfg_zero_enabled,
                zero_init_first_step=self.zero_init_first_step,
            )[0]
            p.sd_model.forge_objects.unet = patched_unet
            p.extra_generation_params["CFG-Zero Enabled"] = self.cfg_zero_enabled
            p.extra_generation_params["CFG-Zero Init First Step"] = self.zero_init_first_step
            logging.debug("CFG-Zero: Patch applied from Guidance Pack.")

        if self.fdg_enabled:
            patched_unet = self.fdg_node.patch(
                p.sd_model.forge_objects.unet,
                fdg_enabled=self.fdg_enabled,
                w_low=self.fdg_w_low,
                w_high=self.fdg_w_high,
                fdg_levels=int(self.fdg_levels),
            )[0]
            p.sd_model.forge_objects.unet = patched_unet
            p.extra_generation_params["FDG Enabled"] = self.fdg_enabled
            p.extra_generation_params["FDG w_low"] = self.fdg_w_low
            p.extra_generation_params["FDG w_high"] = self.fdg_w_high
            p.extra_generation_params["FDG Levels"] = int(self.fdg_levels)
            logging.debug(
                "FDG: Patch applied from Guidance Pack. Enabled=%s, w_low=%s, w_high=%s, levels=%s",
                self.fdg_enabled,
                self.fdg_w_low,
                self.fdg_w_high,
                self.fdg_levels,
            )

        if self.zeresfdg_enabled:
            patched_unet = self.zeresfdg_node.patch(
                p.sd_model.forge_objects.unet,
                zeresfdg_enabled=self.zeresfdg_enabled,
                w_low=self.zeresfdg_w_low,
                w_high=self.zeresfdg_w_high,
                alpha=self.zeresfdg_alpha,
                tau_lo=self.zeresfdg_tau_lo,
                tau_hi=self.zeresfdg_tau_hi,
                beta=self.zeresfdg_beta,
                controller_enabled=self.zeresfdg_controller_enabled,
            )[0]
            p.sd_model.forge_objects.unet = patched_unet
            p.extra_generation_params["ZeResFDG Enabled"] = self.zeresfdg_enabled
            p.extra_generation_params["ZeResFDG λ_l"] = self.zeresfdg_w_low
            p.extra_generation_params["ZeResFDG λ_h"] = self.zeresfdg_w_high
            p.extra_generation_params["ZeResFDG α"] = self.zeresfdg_alpha
            p.extra_generation_params["ZeResFDG τ_lo"] = self.zeresfdg_tau_lo
            p.extra_generation_params["ZeResFDG τ_hi"] = self.zeresfdg_tau_hi
            p.extra_generation_params["ZeResFDG β"] = self.zeresfdg_beta
            p.extra_generation_params["ZeResFDG Controller"] = self.zeresfdg_controller_enabled
            logging.debug(
                "ZeResFDG: Patch applied from Guidance Pack. Enabled=%s, controller=%s, w_low=%s, w_high=%s, alpha=%s, tau_lo=%s, tau_hi=%s, beta=%s",
                self.zeresfdg_enabled,
                self.zeresfdg_controller_enabled,
                self.zeresfdg_w_low,
                self.zeresfdg_w_high,
                self.zeresfdg_alpha,
                self.zeresfdg_tau_lo,
                self.zeresfdg_tau_hi,
                self.zeresfdg_beta,
            )

        if self.s2_guidance_enabled:
            patched_unet = self.s2_guidance_node.patch(
                p.sd_model.forge_objects.unet,
                s2_guidance_enabled=self.s2_guidance_enabled,
                s2_scale_omega=self.s2_scale_omega,
                s2_drop_ratio=self.s2_drop_ratio,
            )[0]
            p.sd_model.forge_objects.unet = patched_unet
            p.extra_generation_params["S2-Guidance Enabled"] = self.s2_guidance_enabled
            p.extra_generation_params["S2-Guidance Omega"] = self.s2_scale_omega
            p.extra_generation_params["S2-Guidance Drop Ratio"] = self.s2_drop_ratio
            logging.debug(
                "S2-Guidance: Patch applied from Guidance Pack. Enabled=%s, omega=%s, drop_ratio=%s",
                self.s2_guidance_enabled,
                self.s2_scale_omega,
                self.s2_drop_ratio,
            )

        if self.tpso_enabled:
            patched_unet = self.tpso_node.patch(
                p.sd_model.forge_objects.unet,
                p=p,
                tpso_enabled=self.tpso_enabled,
                tpso_steps=self.tpso_steps,
                tpso_lr=self.tpso_lr,
                tpso_lambda=self.tpso_lambda,
                tpso_r=self.tpso_r,
                tpso_kappa=self.tpso_kappa,
            )[0]
            p.sd_model.forge_objects.unet = patched_unet
            p.extra_generation_params["TPSO Enabled"] = self.tpso_enabled
            p.extra_generation_params["TPSO Steps"] = self.tpso_steps
            p.extra_generation_params["TPSO LR"] = self.tpso_lr
            p.extra_generation_params["TPSO Lambda"] = self.tpso_lambda
            p.extra_generation_params["TPSO r"] = self.tpso_r
            logging.debug(
                "TPSO: Patch applied. Enabled=%s, steps=%s, r=%s",
                self.tpso_enabled,
                self.tpso_steps,
                self.tpso_r,
            )

        pipeline = None
        needs_qsilk_pipeline = self.qsilk_enabled or hasattr(p.sd_model.forge_objects.unet, "_guidance_pipeline")
        if needs_qsilk_pipeline:
            pipeline = ensure_guidance_pipeline(p.sd_model.forge_objects.unet)

        if self.qsilk_enabled and pipeline is not None:
            pipeline.add_modifier(
                "qsilk",
                make_qsilk_modifier(
                    micro_q_low=self.micro_q_low,
                    micro_q_high=self.micro_q_high,
                    micro_alpha=self.micro_alpha,
                    micro_vp_mix=self.qsilk_vp_mix,
                    per_channel=self.qsilk_per_channel,
                    enable_late_weighting=self.qsilk_late_weighting,
                    use_aqclip=self.qsilk_use_aqclip,
                    tile_size=int(self.qsilk_tile_size),
                    stride=int(self.qsilk_stride),
                    aqclip_alpha=self.qsilk_aqclip_alpha,
                    ema_beta=self.qsilk_ema_beta,
                ),
            )
            p.extra_generation_params["QSilk Enabled"] = self.qsilk_enabled
            p.extra_generation_params["QSilk micro_q_low"] = self.micro_q_low
            p.extra_generation_params["QSilk micro_q_high"] = self.micro_q_high
            p.extra_generation_params["QSilk micro_alpha"] = self.micro_alpha
            p.extra_generation_params["QSilk micro_vp_mix"] = self.qsilk_vp_mix
            p.extra_generation_params["QSilk per_channel"] = self.qsilk_per_channel
            p.extra_generation_params["QSilk late_weighting"] = self.qsilk_late_weighting
            p.extra_generation_params["QSilk use_aqclip"] = self.qsilk_use_aqclip
            p.extra_generation_params["QSilk tile_size"] = int(self.qsilk_tile_size)
            p.extra_generation_params["QSilk stride"] = int(self.qsilk_stride)
            p.extra_generation_params["QSilk aqclip_alpha"] = self.qsilk_aqclip_alpha
            p.extra_generation_params["QSilk ema_beta"] = self.qsilk_ema_beta
            logging.debug(
                "QSilk: Applied with micrograin (%s, %s, %s, vp=%s, per_channel=%s, late_weighting=%s) and AQClip (%s, tile=%s, stride=%s, alpha=%s, ema=%s)",
                self.micro_q_low,
                self.micro_q_high,
                self.micro_alpha,
                self.qsilk_vp_mix,
                self.qsilk_per_channel,
                self.qsilk_late_weighting,
                self.qsilk_use_aqclip,
                self.qsilk_tile_size,
                self.qsilk_stride,
                self.qsilk_aqclip_alpha,
                self.qsilk_ema_beta,
            )
        elif pipeline is not None and "qsilk" in pipeline.modifiers:
            pipeline.modifiers.pop("qsilk", None)

        return


def set_guidance_value(p, x, xs, *, feature: str, field: str):
    if not hasattr(p, "_guidance_xyz"):
        p._guidance_xyz = {}
    feature_map = p._guidance_xyz.setdefault(feature, {})
    feature_map[field] = str(x)


def make_guidance_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None:
        return

    prefixes = ["(CFG-Zero)", "(FDG)", "(ZeResFDG)", "(S2-Guidance)", "(QSilk)", "(TPSO)"]
    if any(opt.label.startswith(prefix) for opt in xyz_grid.axis_options for prefix in prefixes):
        return

    options = [
        xyz_grid.AxisOption(
            label="(CFG-Zero) Enabled",
            type=str,
            apply=partial(set_guidance_value, feature="cfg_zero", field="cfg_zero_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(CFG-Zero) Zero Init",
            type=str,
            apply=partial(set_guidance_value, feature="cfg_zero", field="zero_init"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(FDG) Enabled",
            type=str,
            apply=partial(set_guidance_value, feature="fdg", field="fdg_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(label="(FDG) w_low", type=float, apply=partial(set_guidance_value, feature="fdg", field="w_low")),
        xyz_grid.AxisOption(label="(FDG) w_high", type=float, apply=partial(set_guidance_value, feature="fdg", field="w_high")),
        xyz_grid.AxisOption(
            label="(FDG) Levels", type=int, apply=partial(set_guidance_value, feature="fdg", field="fdg_levels")
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) Enabled",
            type=str,
            apply=partial(set_guidance_value, feature="zeresfdg", field="zeresfdg_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) Controller",
            type=str,
            apply=partial(set_guidance_value, feature="zeresfdg", field="controller_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) λ_l",
            type=float,
            apply=partial(set_guidance_value, feature="zeresfdg", field="w_low"),
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) λ_h",
            type=float,
            apply=partial(set_guidance_value, feature="zeresfdg", field="w_high"),
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) α",
            type=float,
            apply=partial(set_guidance_value, feature="zeresfdg", field="alpha"),
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) τ_lo",
            type=float,
            apply=partial(set_guidance_value, feature="zeresfdg", field="tau_lo"),
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) τ_hi",
            type=float,
            apply=partial(set_guidance_value, feature="zeresfdg", field="tau_hi"),
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) β",
            type=float,
            apply=partial(set_guidance_value, feature="zeresfdg", field="beta"),
        ),
        xyz_grid.AxisOption(
            label="(S2-Guidance) Enabled",
            type=str,
            apply=partial(set_guidance_value, feature="s2_guidance", field="s2_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(S2-Guidance) Omega",
            type=float,
            apply=partial(set_guidance_value, feature="s2_guidance", field="s2_omega"),
        ),
        xyz_grid.AxisOption(
            label="(S2-Guidance) Drop Ratio",
            type=float,
            apply=partial(set_guidance_value, feature="s2_guidance", field="s2_drop"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) Enabled",
            type=str,
            apply=partial(set_guidance_value, feature="qsilk", field="qsilk_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(QSilk) micro_q_low",
            type=float,
            apply=partial(set_guidance_value, feature="qsilk", field="micro_q_low"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) micro_q_high",
            type=float,
            apply=partial(set_guidance_value, feature="qsilk", field="micro_q_high"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) micro_alpha",
            type=float,
            apply=partial(set_guidance_value, feature="qsilk", field="micro_alpha"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) micro_vp_mix",
            type=float,
            apply=partial(set_guidance_value, feature="qsilk", field="micro_vp_mix"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) per_channel",
            type=str,
            apply=partial(set_guidance_value, feature="qsilk", field="per_channel"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(QSilk) late_weighting",
            type=str,
            apply=partial(set_guidance_value, feature="qsilk", field="late_weighting"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(QSilk) use_aqclip",
            type=str,
            apply=partial(set_guidance_value, feature="qsilk", field="use_aqclip"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(QSilk) tile_size",
            type=int,
            apply=partial(set_guidance_value, feature="qsilk", field="tile_size"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) stride",
            type=int,
            apply=partial(set_guidance_value, feature="qsilk", field="stride"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) aqclip_alpha",
            type=float,
            apply=partial(set_guidance_value, feature="qsilk", field="aqclip_alpha"),
        ),
        xyz_grid.AxisOption(
            label="(QSilk) ema_beta",
            type=float,
            apply=partial(set_guidance_value, feature="qsilk", field="ema_beta"),
        ),
        xyz_grid.AxisOption(
            label="(TPSO) Enabled",
            type=str,
            apply=partial(set_guidance_value, feature="tpso", field="tpso_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(TPSO) Steps",
            type=int,
            apply=partial(set_guidance_value, feature="tpso", field="tpso_steps"),
        ),
        xyz_grid.AxisOption(
            label="(TPSO) LR",
            type=float,
            apply=partial(set_guidance_value, feature="tpso", field="tpso_lr"),
        ),
        xyz_grid.AxisOption(
            label="(TPSO) Lambda",
            type=float,
            apply=partial(set_guidance_value, feature="tpso", field="tpso_lambda"),
        ),
        xyz_grid.AxisOption(
            label="(TPSO) r (Schedule)",
            type=float,
            apply=partial(set_guidance_value, feature="tpso", field="tpso_r"),
        ),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("Guidance Pack: XYZ Grid options registered.")


def on_guidance_pack_before_ui():
    try:
        make_guidance_axis_on_xyz_grid()
    except Exception:
        print(
            f"[-] Guidance Pack Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}",
            file=sys.stderr,
        )


script_callbacks.on_before_ui(on_guidance_pack_before_ui)
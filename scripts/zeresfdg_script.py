import logging
import sys
import traceback
from functools import partial

import gradio as gr
from modules import scripts, script_callbacks

from guidance_utils import clear_generation_params_once, reset_unet_if_needed
from nodes_zeresfdg import ZeResFDGNode


class ZeResFDGScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.zeresfdg_enabled = False
        self.w_low = 0.6
        self.w_high = 1.3
        self.alpha = 0.7
        self.tau_lo = 0.45
        self.tau_hi = 0.6
        self.beta = 0.8
        self.controller_enabled = True
        self.node_instance = ZeResFDGNode()

    sorting_priority = 15.05

    def title(self):
        return "ZeResFDG Guidance (CADE 2.5)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Enable ZeResFDG guidance with spectral EMA switching between CFGZeroFD and RescaleFDG modes.")
            zeresfdg_enabled = gr.Checkbox(label="Enable ZeResFDG", value=self.zeresfdg_enabled)
            controller_enabled = gr.Checkbox(label="Enable Spectral Controller", value=self.controller_enabled)
            w_low = gr.Slider(label="λ_l (Low-Frequency Gain)", minimum=0.0, maximum=10.0, step=0.05, value=self.w_low)
            w_high = gr.Slider(label="λ_h (High-Frequency Gain)", minimum=0.0, maximum=10.0, step=0.05, value=self.w_high)
            alpha = gr.Slider(label="α (Rescale Blend)", minimum=0.0, maximum=1.0, step=0.01, value=self.alpha)
            tau_lo = gr.Slider(label="τ_lo (EMA Low Threshold)", minimum=0.0, maximum=1.0, step=0.01, value=self.tau_lo)
            tau_hi = gr.Slider(label="τ_hi (EMA High Threshold)", minimum=0.0, maximum=1.0, step=0.01, value=self.tau_hi)
            beta = gr.Slider(label="β (EMA Smoothing)", minimum=0.0, maximum=1.0, step=0.01, value=self.beta)

        zeresfdg_enabled.change(lambda x: setattr(self, "zeresfdg_enabled", x), inputs=[zeresfdg_enabled], outputs=None)
        controller_enabled.change(lambda x: setattr(self, "controller_enabled", x), inputs=[controller_enabled], outputs=None)
        w_low.change(lambda x: setattr(self, "w_low", x), inputs=[w_low], outputs=None)
        w_high.change(lambda x: setattr(self, "w_high", x), inputs=[w_high], outputs=None)
        alpha.change(lambda x: setattr(self, "alpha", x), inputs=[alpha], outputs=None)
        tau_lo.change(lambda x: setattr(self, "tau_lo", x), inputs=[tau_lo], outputs=None)
        tau_hi.change(lambda x: setattr(self, "tau_hi", x), inputs=[tau_hi], outputs=None)
        beta.change(lambda x: setattr(self, "beta", x), inputs=[beta], outputs=None)

        self.ui_controls = [
            zeresfdg_enabled,
            controller_enabled,
            w_low,
            w_high,
            alpha,
            tau_lo,
            tau_hi,
            beta,
        ]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 8:
            (
                self.zeresfdg_enabled,
                self.controller_enabled,
                self.w_low,
                self.w_high,
                self.alpha,
                self.tau_lo,
                self.tau_hi,
                self.beta,
            ) = args[:8]
        else:
            logging.warning("ZeResFDG: Not enough arguments provided from UI.")

        xyz_settings = getattr(p, "_zeresfdg_xyz", {})
        if "zeresfdg_enabled" in xyz_settings:
            self.zeresfdg_enabled = str(xyz_settings["zeresfdg_enabled"]).lower() == "true"
        if "controller_enabled" in xyz_settings:
            self.controller_enabled = str(xyz_settings["controller_enabled"]).lower() == "true"
        if "w_low" in xyz_settings:
            self.w_low = float(xyz_settings["w_low"])
        if "w_high" in xyz_settings:
            self.w_high = float(xyz_settings["w_high"])
        if "alpha" in xyz_settings:
            self.alpha = float(xyz_settings["alpha"])
        if "tau_lo" in xyz_settings:
            self.tau_lo = float(xyz_settings["tau_lo"])
        if "tau_hi" in xyz_settings:
            self.tau_hi = float(xyz_settings["tau_hi"])
        if "beta" in xyz_settings:
            self.beta = float(xyz_settings["beta"])

        restored = reset_unet_if_needed(p)
        if restored:
            clear_generation_params_once(p)

        if not self.zeresfdg_enabled:
            logging.debug("ZeResFDG: disabled. No patch applied.")
            return

        patched_unet = self.node_instance.patch(
            p.sd_model.forge_objects.unet,
            zeresfdg_enabled=self.zeresfdg_enabled,
            w_low=self.w_low,
            w_high=self.w_high,
            alpha=self.alpha,
            tau_lo=self.tau_lo,
            tau_hi=self.tau_hi,
            beta=self.beta,
            controller_enabled=self.controller_enabled,
        )[0]

        p.sd_model.forge_objects.unet = patched_unet
        p.extra_generation_params["ZeResFDG Enabled"] = self.zeresfdg_enabled
        p.extra_generation_params["ZeResFDG λ_l"] = self.w_low
        p.extra_generation_params["ZeResFDG λ_h"] = self.w_high
        p.extra_generation_params["ZeResFDG α"] = self.alpha
        p.extra_generation_params["ZeResFDG τ_lo"] = self.tau_lo
        p.extra_generation_params["ZeResFDG τ_hi"] = self.tau_hi
        p.extra_generation_params["ZeResFDG β"] = self.beta
        p.extra_generation_params["ZeResFDG Controller"] = self.controller_enabled
        logging.debug(
            "ZeResFDG: Patch applied. Enabled=%s, controller=%s, w_low=%s, w_high=%s, alpha=%s, tau_lo=%s, tau_hi=%s, beta=%s",
            self.zeresfdg_enabled,
            self.controller_enabled,
            self.w_low,
            self.w_high,
            self.alpha,
            self.tau_lo,
            self.tau_hi,
            self.beta,
        )
        return


def zeresfdg_set_value(p, x, xs, *, field: str):
    if not hasattr(p, "_zeresfdg_xyz"):
        p._zeresfdg_xyz = {}
    p._zeresfdg_xyz[field] = str(x)


def make_zeresfdg_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None:
        return

    if any(x.label.startswith("(ZeResFDG)") for x in xyz_grid.axis_options):
        return

    options = [
        xyz_grid.AxisOption(
            label="(ZeResFDG) Enabled",
            type=str,
            apply=partial(zeresfdg_set_value, field="zeresfdg_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(ZeResFDG) Controller",
            type=str,
            apply=partial(zeresfdg_set_value, field="controller_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(label="(ZeResFDG) λ_l", type=float, apply=partial(zeresfdg_set_value, field="w_low")),
        xyz_grid.AxisOption(label="(ZeResFDG) λ_h", type=float, apply=partial(zeresfdg_set_value, field="w_high")),
        xyz_grid.AxisOption(label="(ZeResFDG) α", type=float, apply=partial(zeresfdg_set_value, field="alpha")),
        xyz_grid.AxisOption(label="(ZeResFDG) τ_lo", type=float, apply=partial(zeresfdg_set_value, field="tau_lo")),
        xyz_grid.AxisOption(label="(ZeResFDG) τ_hi", type=float, apply=partial(zeresfdg_set_value, field="tau_hi")),
        xyz_grid.AxisOption(label="(ZeResFDG) β", type=float, apply=partial(zeresfdg_set_value, field="beta")),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("ZeResFDG: XYZ Grid options registered.")


def on_zeresfdg_before_ui():
    try:
        make_zeresfdg_axis_on_xyz_grid()
    except Exception:
        print(f"[-] ZeResFDG Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}", file=sys.stderr)


script_callbacks.on_before_ui(on_zeresfdg_before_ui)

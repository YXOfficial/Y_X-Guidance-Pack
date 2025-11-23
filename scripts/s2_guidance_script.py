import logging
import sys
import traceback
from functools import partial

import gradio as gr
from modules import scripts, script_callbacks

from guidance_utils import clear_generation_params_once, reset_unet_if_needed
from nodes_s2_guidance import S2GuidanceNode


class S2GuidanceScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.s2_guidance_enabled = False
        self.s2_scale_omega = 0.25
        self.s2_drop_ratio = 0.1
        self.node_instance = S2GuidanceNode()

    sorting_priority = 15.0

    def title(self):
        return "S²-Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Enable and configure S²-Guidance. Increases generation time.")
            s2_guidance_enabled = gr.Checkbox(label="Enable S²-Guidance", value=self.s2_guidance_enabled)
            s2_scale_omega = gr.Slider(
                label="S² Scale (ω)", minimum=0.0, maximum=2.0, step=0.05, value=self.s2_scale_omega
            )
            s2_drop_ratio = gr.Slider(
                label="Block Drop Ratio", minimum=0.0, maximum=0.5, step=0.01, value=self.s2_drop_ratio,
                info="Tỷ lệ khối UNet/DiT bị bỏ qua. Bài báo đề xuất ~0.1 (10%)"
            )

        s2_guidance_enabled.change(lambda x: setattr(self, "s2_guidance_enabled", x), inputs=[s2_guidance_enabled], outputs=None)
        s2_scale_omega.change(lambda x: setattr(self, "s2_scale_omega", x), inputs=[s2_scale_omega], outputs=None)
        s2_drop_ratio.change(lambda x: setattr(self, "s2_drop_ratio", x), inputs=[s2_drop_ratio], outputs=None)

        self.ui_controls = [s2_guidance_enabled, s2_scale_omega, s2_drop_ratio]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 3:
            self.s2_guidance_enabled, self.s2_scale_omega, self.s2_drop_ratio = args[:3]
        else:
            logging.warning("S2-Guidance: Not enough arguments provided from UI.")

        xyz_settings = getattr(p, "_s2_guidance_xyz", {})
        if "s2_enabled" in xyz_settings:
            self.s2_guidance_enabled = str(xyz_settings["s2_enabled"]).lower() == "true"
        if "s2_omega" in xyz_settings:
            self.s2_scale_omega = float(xyz_settings["s2_omega"])
        if "s2_drop" in xyz_settings:
            self.s2_drop_ratio = float(xyz_settings["s2_drop"])

        restored = reset_unet_if_needed(p)
        if restored:
            clear_generation_params_once(p)

        if not self.s2_guidance_enabled:
            logging.debug("S2-Guidance: disabled. No patch applied.")
            return

        patched_unet = self.node_instance.patch(
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
            f"S2-Guidance: Patch applied. Enabled={self.s2_guidance_enabled}, omega={self.s2_scale_omega}, drop_ratio={self.s2_drop_ratio}"
        )
        return


def s2_set_value(p, x, xs, *, field: str):
    if not hasattr(p, "_s2_guidance_xyz"):
        p._s2_guidance_xyz = {}
    p._s2_guidance_xyz[field] = str(x)


def make_s2_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None:
        return

    if any(x.label.startswith("(S2-Guidance)") for x in xyz_grid.axis_options):
        return

    options = [
        xyz_grid.AxisOption(
            label="(S2-Guidance) Enabled",
            type=str,
            apply=partial(s2_set_value, field="s2_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(label="(S2-Guidance) Omega", type=float, apply=partial(s2_set_value, field="s2_omega")),
        xyz_grid.AxisOption(label="(S2-Guidance) Drop Ratio", type=float, apply=partial(s2_set_value, field="s2_drop")),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("S2-Guidance: XYZ Grid options registered.")


def on_s2_before_ui():
    try:
        make_s2_axis_on_xyz_grid()
    except Exception:
        print(f"[-] S2-Guidance Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}", file=sys.stderr)


script_callbacks.on_before_ui(on_s2_before_ui)

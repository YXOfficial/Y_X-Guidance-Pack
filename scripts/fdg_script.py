import logging
import sys
import traceback
from functools import partial

import gradio as gr
from modules import scripts, script_callbacks

from guidance_utils import clear_generation_params_once, reset_unet_if_needed
from nodes_fdg import FDGNode


class FDGScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.fdg_enabled = False
        self.w_low = 1.0
        self.w_high = 1.0
        self.fdg_levels = 3
        self.node_instance = FDGNode()

    sorting_priority = 15.1

    def title(self):
        return "Frequency-Decoupled Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Configure Frequency-Decoupled Guidance.")
            fdg_enabled = gr.Checkbox(label="Enable FDG", value=self.fdg_enabled)
            w_low = gr.Slider(
                label="w_low (Low-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.w_low
            )
            w_high = gr.Slider(
                label="w_high (High-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.w_high
            )
            fdg_levels = gr.Slider(label="FDG Pyramid Levels", minimum=2, maximum=8, step=1, value=self.fdg_levels)

        fdg_enabled.change(lambda x: setattr(self, "fdg_enabled", x), inputs=[fdg_enabled], outputs=None)
        w_low.change(lambda x: setattr(self, "w_low", x), inputs=[w_low], outputs=None)
        w_high.change(lambda x: setattr(self, "w_high", x), inputs=[w_high], outputs=None)
        fdg_levels.change(lambda x: setattr(self, "fdg_levels", x), inputs=[fdg_levels], outputs=None)

        self.ui_controls = [fdg_enabled, w_low, w_high, fdg_levels]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 4:
            self.fdg_enabled, self.w_low, self.w_high, self.fdg_levels = args[:4]
        else:
            logging.warning("FDG: Not enough arguments provided from UI.")

        xyz_settings = getattr(p, "_fdg_xyz", {})
        if "fdg_enabled" in xyz_settings:
            self.fdg_enabled = str(xyz_settings["fdg_enabled"]).lower() == "true"
        if "w_low" in xyz_settings:
            self.w_low = float(xyz_settings["w_low"])
        if "w_high" in xyz_settings:
            self.w_high = float(xyz_settings["w_high"])
        if "fdg_levels" in xyz_settings:
            self.fdg_levels = int(xyz_settings["fdg_levels"])

        restored = reset_unet_if_needed(p)
        if restored:
            clear_generation_params_once(p)

        if not self.fdg_enabled:
            logging.debug("FDG: disabled. No patch applied.")
            return

        patched_unet = self.node_instance.patch(
            p.sd_model.forge_objects.unet,
            fdg_enabled=self.fdg_enabled,
            w_low=self.w_low,
            w_high=self.w_high,
            fdg_levels=int(self.fdg_levels),
        )[0]

        p.sd_model.forge_objects.unet = patched_unet
        p.extra_generation_params["FDG Enabled"] = self.fdg_enabled
        p.extra_generation_params["FDG w_low"] = self.w_low
        p.extra_generation_params["FDG w_high"] = self.w_high
        p.extra_generation_params["FDG Levels"] = int(self.fdg_levels)
        logging.debug(
            f"FDG: Patch applied. Enabled={self.fdg_enabled}, w_low={self.w_low}, w_high={self.w_high}, levels={self.fdg_levels}"
        )
        return


def fdg_set_value(p, x, xs, *, field: str):
    if not hasattr(p, "_fdg_xyz"):
        p._fdg_xyz = {}
    p._fdg_xyz[field] = str(x)


def make_fdg_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None:
        return

    if any(x.label.startswith("(FDG)") for x in xyz_grid.axis_options):
        return

    options = [
        xyz_grid.AxisOption(
            label="(FDG) Enabled",
            type=str,
            apply=partial(fdg_set_value, field="fdg_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(label="(FDG) w_low", type=float, apply=partial(fdg_set_value, field="w_low")),
        xyz_grid.AxisOption(label="(FDG) w_high", type=float, apply=partial(fdg_set_value, field="w_high")),
        xyz_grid.AxisOption(label="(FDG) Levels", type=int, apply=partial(fdg_set_value, field="fdg_levels")),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("FDG: XYZ Grid options registered.")


def on_fdg_before_ui():
    try:
        make_fdg_axis_on_xyz_grid()
    except Exception:
        print(f"[-] FDG Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}", file=sys.stderr)


script_callbacks.on_before_ui(on_fdg_before_ui)

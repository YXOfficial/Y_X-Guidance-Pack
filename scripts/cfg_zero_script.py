import logging
import sys
import traceback
from functools import partial

import gradio as gr
from modules import scripts, script_callbacks

from guidance_utils import clear_generation_params_once, reset_unet_if_needed
from nodes_cfg_zero import CFGZeroNode


class CFGZeroScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.cfg_zero_enabled = False
        self.zero_init_first_step = False
        self.node_instance = CFGZeroNode()

    sorting_priority = 15.2

    def title(self):
        return "CFG-Zero Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Enable CFG-Zero guidance adjustments.")
            cfg_zero_enabled = gr.Checkbox(label="Enable CFG-Zero", value=self.cfg_zero_enabled)
            zero_init_first_step = gr.Checkbox(
                label="Zero Init First Step (Experimental)", value=self.zero_init_first_step
            )

        cfg_zero_enabled.change(lambda x: setattr(self, "cfg_zero_enabled", x), inputs=[cfg_zero_enabled], outputs=None)
        zero_init_first_step.change(
            lambda x: setattr(self, "zero_init_first_step", x), inputs=[zero_init_first_step], outputs=None
        )

        self.ui_controls = [cfg_zero_enabled, zero_init_first_step]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 2:
            self.cfg_zero_enabled, self.zero_init_first_step = args[:2]
        else:
            logging.warning("CFG-Zero: Not enough arguments provided from UI.")

        xyz_settings = getattr(p, "_cfg_zero_xyz", {})
        if "cfg_zero_enabled" in xyz_settings:
            self.cfg_zero_enabled = str(xyz_settings["cfg_zero_enabled"]).lower() == "true"
        if "zero_init" in xyz_settings:
            self.zero_init_first_step = str(xyz_settings["zero_init"]).lower() == "true"

        restored = reset_unet_if_needed(p)
        if restored:
            clear_generation_params_once(p)

        if not self.cfg_zero_enabled:
            logging.debug("CFG-Zero: disabled. No patch applied.")
            return

        patched_unet = self.node_instance.patch(
            p.sd_model.forge_objects.unet,
            cfg_zero_enabled=self.cfg_zero_enabled,
            zero_init_first_step=self.zero_init_first_step,
        )[0]

        p.sd_model.forge_objects.unet = patched_unet
        p.extra_generation_params["CFG-Zero Enabled"] = self.cfg_zero_enabled
        p.extra_generation_params["CFG-Zero Init First Step"] = self.zero_init_first_step
        logging.debug(f"CFG-Zero: Patch applied. Enabled={self.cfg_zero_enabled}")
        return


def cfg_zero_set_value(p, x, xs, *, field: str):
    if not hasattr(p, "_cfg_zero_xyz"):
        p._cfg_zero_xyz = {}
    p._cfg_zero_xyz[field] = str(x)


def make_cfg_zero_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None:
        return

    if any(x.label.startswith("(CFG-Zero)") for x in xyz_grid.axis_options):
        return

    options = [
        xyz_grid.AxisOption(
            label="(CFG-Zero) Enabled",
            type=str,
            apply=partial(cfg_zero_set_value, field="cfg_zero_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            label="(CFG-Zero) Zero Init",
            type=str,
            apply=partial(cfg_zero_set_value, field="zero_init"),
            choices=lambda: ["True", "False"],
        ),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("CFG-Zero: XYZ Grid options registered.")


def on_cfg_zero_before_ui():
    try:
        make_cfg_zero_axis_on_xyz_grid()
    except Exception:
        print(f"[-] CFG-Zero Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}", file=sys.stderr)


script_callbacks.on_before_ui(on_cfg_zero_before_ui)

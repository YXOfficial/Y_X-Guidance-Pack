import logging
import sys
import traceback
import pkgutil
import importlib
import os
from functools import partial

import gradio as gr
from modules import scripts, script_callbacks

from yx_guidance_utils import (
    clear_generation_params_once,
    reset_unet_if_needed,
)

# Ensure the scripts directory is in path (common issue in some setups)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import registry
from guidance_pack.registry import get_processors

# Dynamically import all processors
try:
    import guidance_pack.processors
    package = guidance_pack.processors
    prefix = package.__name__ + "."
    for _, name, _ in pkgutil.iter_modules(package.__path__, prefix):
        importlib.import_module(name)
except Exception as e:
    logging.error(f"Guidance Pack: Error loading processors: {e}")
    traceback.print_exc()


class GuidancePackScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.processors = get_processors()
        self.arg_counts = []

    sorting_priority = 15.0

    def title(self):
        return "Guidance Pack"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        self.arg_counts = []
        ui_components = []
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Unified UI for Guidance Methods.")
            with gr.Tabs():
                for processor in self.processors:
                    try:
                        components = processor.create_ui()
                        ui_components.extend(components)
                        self.arg_counts.append(len(components))
                    except Exception as e:
                        logging.error(f"Guidance Pack: Error creating UI for {processor.name()}: {e}")
                        self.arg_counts.append(0)
        return ui_components

    def process_before_every_sampling(self, p, *args, **kwargs):
        # Global cleanup
        restored = reset_unet_if_needed(p)
        if restored:
            clear_generation_params_once(p)

        current_arg_index = 0
        for i, processor in enumerate(self.processors):
            if i < len(self.arg_counts):
                count = self.arg_counts[i]
                if count == 0:
                    continue
                
                processor_args = args[current_arg_index : current_arg_index + count]
                current_arg_index += count
                
                try:
                    processor.process(p, *processor_args)
                except Exception as e:
                     logging.error(f"Guidance Pack: Error processing {processor.name()}: {e}")
                     traceback.print_exc()
            else:
                 logging.warning(f"Guidance Pack: Argument count mismatch for {processor.name()}.")

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

    for processor in get_processors():
         try:
            processor.register_xyz(xyz_grid, set_guidance_value)
         except Exception as e:
            logging.error(f"Guidance Pack: Error registering XYZ for {processor.name()}: {e}")

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
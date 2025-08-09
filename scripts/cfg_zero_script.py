# --- START OF FILE cfg_zero_script.py ---
import logging
import sys
import traceback
import gradio as gr
from modules import scripts, script_callbacks
from functools import partial
from typing import Any

from CFGZERO.nodes_cfg_zero import CFGZeroNode

class CFGZeroScript(scripts.Script):
    def __init__(self):
        super().__init__()
        # --- Khởi tạo giá trị mặc định cho tất cả các tùy chọn ---
        self.cfg_zero_enabled = False
        self.zero_init_first_step = False
        self.fdg_enabled = False
        self.w_low = 1.0
        self.w_high = 1.0
        self.fdg_levels = 3
        
        self.node_instance = CFGZeroNode()

    sorting_priority = 15.2

    def title(self):
        return "CFG-Zero / FDG Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Bật và cấu hình CFG-Zero và/hoặc Frequency-Decoupled Guidance (FDG).")
            
            # --- Nhóm CFG-Zero ---
            with gr.Group():
                gr.Markdown("### CFG-Zero Settings")
                cfg_zero_enabled = gr.Checkbox(label="Enable CFG-Zero", value=self.cfg_zero_enabled)
                zero_init_first_step = gr.Checkbox(label="Zero Init First Step (Experimental)", value=self.zero_init_first_step)

            # --- Nhóm FDG ---
            with gr.Group():
                gr.Markdown("### Frequency-Decoupled Guidance (FDG) Settings")
                fdg_enabled = gr.Checkbox(label="Enable FDG (Overrides CFG Scale)", value=self.fdg_enabled)
                w_low = gr.Slider(label="w_low (Low-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.w_low)
                w_high = gr.Slider(label="w_high (High-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.w_high)
                fdg_levels = gr.Slider(label="FDG Pyramid Levels", minimum=2, maximum=8, step=1, value=self.fdg_levels)
        
        # --- Kết nối các hàm `change` đơn giản ---
        cfg_zero_enabled.change(lambda x: setattr(self, 'cfg_zero_enabled', x), inputs=[cfg_zero_enabled], outputs=None)
        zero_init_first_step.change(lambda x: setattr(self, 'zero_init_first_step', x), inputs=[zero_init_first_step], outputs=None)
        fdg_enabled.change(lambda x: setattr(self, 'fdg_enabled', x), inputs=[fdg_enabled], outputs=None)
        w_low.change(lambda x: setattr(self, 'w_low', x), inputs=[w_low], outputs=None)
        w_high.change(lambda x: setattr(self, 'w_high', x), inputs=[w_high], outputs=None)
        fdg_levels.change(lambda x: setattr(self, 'fdg_levels', x), inputs=[fdg_levels], outputs=None)
        
        self.ui_controls = [cfg_zero_enabled, zero_init_first_step, fdg_enabled, w_low, w_high, fdg_levels]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 6:
            (self.cfg_zero_enabled, self.zero_init_first_step, 
             self.fdg_enabled, self.w_low, self.w_high, self.fdg_levels) = args[:6]
        else:
            logging.warning("CFG-Zero/FDG: Not enough arguments provided.")

        # Xử lý XYZ Grid
        xyz_settings = getattr(p, "_cfg_zero_xyz", {})
        if "cfg_zero_enabled" in xyz_settings: self.cfg_zero_enabled = str(xyz_settings["cfg_zero_enabled"]).lower() == "true"
        if "zero_init" in xyz_settings: self.zero_init_first_step = str(xyz_settings["zero_init"]).lower() == "true"
        if "fdg_enabled" in xyz_settings: self.fdg_enabled = str(xyz_settings["fdg_enabled"]).lower() == "true"
        if "w_low" in xyz_settings: self.w_low = float(xyz_settings["w_low"])
        if "w_high" in xyz_settings: self.w_high = float(xyz_settings["w_high"])
        if "fdg_levels" in xyz_settings: self.fdg_levels = int(xyz_settings["fdg_levels"])

        # Khôi phục unet gốc trước khi patch
        if hasattr(p, '_original_unet_before_custom_guidance'):
            p.sd_model.forge_objects.unet = p._original_unet_before_custom_guidance.clone()
        else:
            p._original_unet_before_custom_guidance = p.sd_model.forge_objects.unet.clone()

        # Dọn dẹp metadata trước
        p.extra_generation_params.pop("CFG-Zero Enabled", None)
        p.extra_generation_params.pop("CFG-Zero Init First Step", None)
        p.extra_generation_params.pop("FDG Enabled", None)
        p.extra_generation_params.pop("FDG w_low", None)
        p.extra_generation_params.pop("FDG w_high", None)
        p.extra_generation_params.pop("FDG Levels", None)
        
        # Nếu không có gì được bật, không patch và thoát
        if not self.cfg_zero_enabled and not self.fdg_enabled:
            logging.debug("CFG-Zero/FDG: Both disabled. No patch applied.")
            return

        # Gọi hàm patch với tất cả các tham số
        patched_unet = self.node_instance.patch(
            p.sd_model.forge_objects.unet,
            cfg_zero_enabled=self.cfg_zero_enabled,
            zero_init_first_step=self.zero_init_first_step,
            fdg_enabled=self.fdg_enabled,
            w_low=self.w_low,
            w_high=self.w_high,
            fdg_levels=int(self.fdg_levels)
        )[0]
        
        p.sd_model.forge_objects.unet = patched_unet

        # Cập nhật metadata nếu cần
        if self.cfg_zero_enabled:
            p.extra_generation_params["CFG-Zero Enabled"] = self.cfg_zero_enabled
            p.extra_generation_params["CFG-Zero Init First Step"] = self.zero_init_first_step
        if self.fdg_enabled:
            p.extra_generation_params["FDG Enabled"] = self.fdg_enabled
            p.extra_generation_params["FDG w_low"] = self.w_low
            p.extra_generation_params["FDG w_high"] = self.w_high
            p.extra_generation_params["FDG Levels"] = int(self.fdg_levels)
        
        logging.debug(f"CFG-Zero/FDG: Patch applied. CFG-Zero: {self.cfg_zero_enabled}, FDG: {self.fdg_enabled}")
        return

# --- XYZ Grid Integration (Cần mở rộng) ---
def custom_guidance_set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_cfg_zero_xyz"):
        p._cfg_zero_xyz = {}
    p._cfg_zero_xyz[field] = str(x)

def make_custom_guidance_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None: return

    if any(x.label.startswith("(CustomGuidance)") for x in xyz_grid.axis_options):
        return
        
    options = [
        xyz_grid.AxisOption(label="(CustomGuidance) CFG-Zero Enabled", type=str, apply=partial(custom_guidance_set_value, field="cfg_zero_enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) Zero Init", type=str, apply=partial(custom_guidance_set_value, field="zero_init"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG Enabled", type=str, apply=partial(custom_guidance_set_value, field="fdg_enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG w_low", type=float, apply=partial(custom_guidance_set_value, field="w_low")),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG w_high", type=float, apply=partial(custom_guidance_set_value, field="w_high")),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG Levels", type=int, apply=partial(custom_guidance_set_value, field="fdg_levels")),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("Custom Guidance (CFG-Zero/FDG): XYZ Grid options registered.")

def on_custom_guidance_before_ui():
    try:
        make_custom_guidance_axis_on_xyz_grid()
    except Exception:
        print(f"[-] Custom Guidance Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}", file=sys.stderr)

script_callbacks.on_before_ui(on_custom_guidance_before_ui)
# --- END OF FILE cfg_zero_script.py ---

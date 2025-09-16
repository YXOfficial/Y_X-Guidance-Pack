# --- START OF FILE cfg_zero_script (4).py ---
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
        self.s2_guidance_enabled = False
        self.s2_scale_omega = 0.25
        self.s2_drop_ratio = 0.1
        
        self.node_instance = CFGZeroNode()

    sorting_priority = 15.2

    def title(self):
        return "Custom Guidance (CFG-Zero/FDG/S²)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Bật và cấu hình các phương pháp guidance nâng cao. Chúng có thể được kết hợp.")
            with gr.Group():
                gr.Markdown("### Mahiro Gating")
                mahiro_enabled = gr.Checkbox(label="Enable Mahiro gating", value=getattr(self, "mahiro_enabled", False))
            
            with gr.Group():
                gr.Markdown("### S²-Guidance Settings (Experimental)")
                gr.Markdown("Dựa trên bài báo 'S²-Guidance'. Giảm thiểu các lỗi do mô hình không chắc chắn. **Làm tăng đáng kể thời gian tạo ảnh.**")
                s2_guidance_enabled = gr.Checkbox(label="Enable S²-Guidance", value=self.s2_guidance_enabled)
                s2_scale_omega = gr.Slider(label="S² Scale (ω)", minimum=0.0, maximum=2.0, step=0.05, value=self.s2_scale_omega)
                s2_drop_ratio = gr.Slider(label="Block Drop Ratio", minimum=0.0, maximum=0.5, step=0.01, value=self.s2_drop_ratio, info="Tỷ lệ khối UNet/DiT bị bỏ qua. Bài báo đề xuất ~0.1 (10%)")
            
            with gr.Group():
                gr.Markdown("### CFG-Zero Settings")
                cfg_zero_enabled = gr.Checkbox(label="Enable CFG-Zero", value=self.cfg_zero_enabled)
                zero_init_first_step = gr.Checkbox(label="Zero Init First Step (Experimental)", value=self.zero_init_first_step)

            with gr.Group():
                gr.Markdown("### Frequency-Decoupled Guidance (FDG) Settings")
                fdg_enabled = gr.Checkbox(label="Enable FDG", value=self.fdg_enabled)
                w_low = gr.Slider(label="w_low (Low-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.w_low)
                w_high = gr.Slider(label="w_high (High-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.w_high)
                fdg_levels = gr.Slider(label="FDG Pyramid Levels", minimum=2, maximum=8, step=1, value=self.fdg_levels)
        
        cfg_zero_enabled.change(lambda x: setattr(self, 'cfg_zero_enabled', x), inputs=[cfg_zero_enabled], outputs=None)
        zero_init_first_step.change(lambda x: setattr(self, 'zero_init_first_step', x), inputs=[zero_init_first_step], outputs=None)
        fdg_enabled.change(lambda x: setattr(self, 'fdg_enabled', x), inputs=[fdg_enabled], outputs=None)
        w_low.change(lambda x: setattr(self, 'w_low', x), inputs=[w_low], outputs=None)
        w_high.change(lambda x: setattr(self, 'w_high', x), inputs=[w_high], outputs=None)
        fdg_levels.change(lambda x: setattr(self, 'fdg_levels', x), inputs=[fdg_levels], outputs=None)
        mahiro_enabled.change(lambda x: setattr(self, 'mahiro_enabled', x), inputs=[mahiro_enabled], outputs=None)
        s2_guidance_enabled.change(lambda x: setattr(self, 's2_guidance_enabled', x), inputs=[s2_guidance_enabled], outputs=None)
        s2_scale_omega.change(lambda x: setattr(self, 's2_scale_omega', x), inputs=[s2_scale_omega], outputs=None)
        s2_drop_ratio.change(lambda x: setattr(self, 's2_drop_ratio', x), inputs=[s2_drop_ratio], outputs=None)
        
        self.ui_controls = [cfg_zero_enabled, zero_init_first_step, fdg_enabled, w_low, w_high, fdg_levels, 
                            s2_guidance_enabled, s2_scale_omega, s2_drop_ratio]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 9:
            (self.cfg_zero_enabled, self.zero_init_first_step, 
             self.fdg_enabled, self.w_low, self.w_high, self.fdg_levels,
             self.s2_guidance_enabled, self.s2_scale_omega, self.s2_drop_ratio, self.mahiro_enabled) = args[:10]
        else:
            logging.warning("Custom Guidance: Not enough arguments provided from UI.")

        # Xử lý XYZ Grid
        xyz_settings = getattr(p, "_custom_guidance_xyz", {})
        if "cfg_zero_enabled" in xyz_settings: self.cfg_zero_enabled = str(xyz_settings["cfg_zero_enabled"]).lower() == "true"
        if "zero_init" in xyz_settings: self.zero_init_first_step = str(xyz_settings["zero_init"]).lower() == "true"
        if "fdg_enabled" in xyz_settings: self.fdg_enabled = str(xyz_settings["fdg_enabled"]).lower() == "true"
        if "w_low" in xyz_settings: self.w_low = float(xyz_settings["w_low"])
        if "w_high" in xyz_settings: self.w_high = float(xyz_settings["w_high"])
        if "fdg_levels" in xyz_settings: self.fdg_levels = int(xyz_settings["fdg_levels"])
        if "s2_enabled" in xyz_settings: self.s2_guidance_enabled = str(xyz_settings["s2_enabled"]).lower() == "true"
        if "s2_omega" in xyz_settings: self.s2_scale_omega = float(xyz_settings["s2_omega"])
        if "s2_drop" in xyz_settings: self.s2_drop_ratio = float(xyz_settings["s2_drop"])

        if hasattr(p, '_original_unet_before_custom_guidance'):
            p.sd_model.forge_objects.unet = p._original_unet_before_custom_guidance.clone()
        else:
            p._original_unet_before_custom_guidance = p.sd_model.forge_objects.unet.clone()

        # Dọn dẹp metadata
        params_to_pop = ["CFG-Zero Enabled", "CFG-Zero Init First Step", "FDG Enabled", "FDG w_low", "FDG w_high", "FDG Levels", "S2-Guidance Enabled", "S2-Guidance Omega", "S2-Guidance Drop Ratio"]
        for param in params_to_pop:
            p.extra_generation_params.pop(param, None)
        
        if not self.cfg_zero_enabled and not self.fdg_enabled and not self.s2_guidance_enabled:
            logging.debug("Custom Guidance: All methods disabled. No patch applied.")
            return

        patched_unet = self.node_instance.patch(
            p.sd_model.forge_objects.unet,
            cfg_zero_enabled=self.cfg_zero_enabled,
            zero_init_first_step=self.zero_init_first_step,
            fdg_enabled=self.fdg_enabled,
            w_low=self.w_low,
            w_high=self.w_high,
            fdg_levels=int(self.fdg_levels),
            s2_guidance_enabled=self.s2_guidance_enabled,
            s2_scale_omega=self.s2_scale_omega,
            s2_drop_ratio=self.s2_drop_ratio
        )[0]
        

# Optional: append Mahiro gating as a second post‑CFG
        # Optional: append Mahiro gating as a second post‑CFG
        if getattr(self, "mahiro_enabled", False):
            import torch, torch.nn.functional as F
            def _mahiro_post(args):
                scale = float(args.get("cond_scale", 1.0))
                C = args["cond_denoised"]
                U = args["uncond_denoised"]
                cfg = args["denoised"]  # current CFG/FDG/S² result
                leap = C * scale
                merge = 0.5 * (leap + cfg)
                def srs(x): return torch.sqrt(x.abs() + 1e-12) * x.sign()
                u_leap = U * scale
                sim = F.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, u_leap.ndim))).mean()
                gate = 2.0 * (sim + 1.0)  # [0,4]
                return (gate * cfg + (4.0 - gate) * leap) / 4.0
            try:
                patched_unet.set_model_sampler_post_cfg_function(_mahiro_post, "mahiro_gate")
            except Exception:
                from ldm_patched.modules.model_patcher import set_model_options_post_cfg_function
                mo = getattr(patched_unet, "model_options", {})
                mo = set_model_options_post_cfg_function(mo, _mahiro_post, disable_cfg1_optimization=True)
                patched_unet.set_model_options(mo)
        p.sd_model.forge_objects.unet = patched_unet

        # Cập nhật metadata
        if self.cfg_zero_enabled:
            p.extra_generation_params["CFG-Zero Enabled"] = self.cfg_zero_enabled
            p.extra_generation_params["CFG-Zero Init First Step"] = self.zero_init_first_step
        if self.fdg_enabled:
            p.extra_generation_params["FDG Enabled"] = self.fdg_enabled
            p.extra_generation_params["FDG w_low"] = self.w_low
            p.extra_generation_params["FDG w_high"] = self.w_high
            p.extra_generation_params["FDG Levels"] = int(self.fdg_levels)
        if self.s2_guidance_enabled:
            p.extra_generation_params["S2-Guidance Enabled"] = self.s2_guidance_enabled
            p.extra_generation_params["S2-Guidance Omega"] = self.s2_scale_omega
            p.extra_generation_params["S2-Guidance Drop Ratio"] = self.s2_drop_ratio
        if getattr(self, "mahiro_enabled", False):
            p.extra_generation_params["Mahiro Enabled"] = True
        
        logging.debug(f"Custom Guidance: Patch applied. CFG-Zero: {self.cfg_zero_enabled}, FDG: {self.fdg_enabled}, S2-Guidance: {self.s2_guidance_enabled}")
        return

def custom_guidance_set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_custom_guidance_xyz"):
        p._custom_guidance_xyz = {}
    p._custom_guidance_xyz[field] = str(x)

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
        xyz_grid.AxisOption(label="(CustomGuidance) S2 Enabled", type=str, apply=partial(custom_guidance_set_value, field="s2_enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) S2 Omega", type=float, apply=partial(custom_guidance_set_value, field="s2_omega")),
        xyz_grid.AxisOption(label="(CustomGuidance) S2 Drop Ratio", type=float, apply=partial(custom_guidance_set_value, field="s2_drop")),
        xyz_grid.AxisOption(label="(CustomGuidance) CFG-Zero Enabled", type=str, apply=partial(custom_guidance_set_value, field="cfg_zero_enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) Zero Init", type=str, apply=partial(custom_guidance_set_value, field="zero_init"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG Enabled", type=str, apply=partial(custom_guidance_set_value, field="fdg_enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG w_low", type=float, apply=partial(custom_guidance_set_value, field="w_low")),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG w_high", type=float, apply=partial(custom_guidance_set_value, field="w_high")),
        xyz_grid.AxisOption(label="(CustomGuidance) FDG Levels", type=int, apply=partial(custom_guidance_set_value, field="fdg_levels")),
    ]
    xyz_grid.axis_options.extend(options)
    logging.info("Custom Guidance: XYZ Grid options registered.")

def on_custom_guidance_before_ui():
    try:
        make_custom_guidance_axis_on_xyz_grid()
    except Exception:
        print(f"[-] Custom Guidance Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}", file=sys.stderr)

script_callbacks.on_before_ui(on_custom_guidance_before_ui)
# --- END OF FILE cfg_zero_script (4).py ---
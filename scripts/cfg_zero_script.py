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
                            s2_guidance_enabled, s2_scale_omega, s2_drop_ratio, mahiro_enabled]
        return self.ui_controls

    def process_before_every_sampling(self, p, *args, **kwargs):
        # Robustly unpack 9 or 10 args (Mahiro added as the last UI control)
        if len(args) >= 10:
            (self.cfg_zero_enabled, self.zero_init_first_step,
             self.fdg_enabled, self.w_low, self.w_high, self.fdg_levels,
             self.s2_guidance_enabled, self.s2_scale_omega, self.s2_drop_ratio, self.mahiro_enabled) = args[:10]
        elif len(args) >= 9:
            (self.cfg_zero_enabled, self.zero_init_first_step,
             self.fdg_enabled, self.w_low, self.w_high, self.fdg_levels,
             self.s2_guidance_enabled, self.s2_scale_omega, self.s2_drop_ratio) = args[:9]
            if not hasattr(self, 'mahiro_enabled'):
                self.mahiro_enabled = False

        # XYZ plot overrides, if present
        xyz_settings = getattr(p, 'xyz_settings', {}) if hasattr(p, 'xyz_settings') else {}
        try:
            if 'cfg_zero' in xyz_settings: self.cfg_zero_enabled = bool(xyz_settings['cfg_zero'])
            if 'zero_init' in xyz_settings: self.zero_init_first_step = bool(xyz_settings['zero_init'])
            if 'fdg' in xyz_settings: self.fdg_enabled = bool(xyz_settings['fdg'])
            if 'w_low' in xyz_settings: self.w_low = float(xyz_settings['w_low'])
            if 'w_high' in xyz_settings: self.w_high = float(xyz_settings['w_high'])
            if 'fdg_levels' in xyz_settings: self.fdg_levels = int(xyz_settings['fdg_levels'])
            if 's2' in xyz_settings: self.s2_guidance_enabled = bool(xyz_settings['s2'])
            if 's2_omega' in xyz_settings: self.s2_scale_omega = float(xyz_settings['s2_omega'])
            if 's2_drop' in xyz_settings: self.s2_drop_ratio = float(xyz_settings['s2_drop'])
            if 'mahiro_enabled' in xyz_settings: self.mahiro_enabled = str(xyz_settings['mahiro_enabled']).lower() == 'true'
        except Exception:
            pass

        # Reset and patch UNet just like the original script
        try:
            if hasattr(p.sd_model, 'forge_objects') and hasattr(p.sd_model.forge_objects, 'unet'):
                unet = p.sd_model.forge_objects.unet
            else:
                unet = p.sd_model
            # original script resets and patches inside process(); we keep compatibility here
        except Exception:
            pass

def process(self, p, *args):
    try:
        # get ModelPatcher UNet
        unet = p.sd_model.forge_objects.unet if hasattr(p.sd_model, 'forge_objects') else p.sd_model

        # apply original CFG-Zero/FDG/S² node patch
        patched_unet, = CFGZeroNode().patch(
            unet,
            cfg_zero_enabled=getattr(self, "cfg_zero_enabled", False),
            zero_init_first_step=getattr(self, "zero_init_first_step", False),
            fdg_enabled=getattr(self, "fdg_enabled", False),
            w_low=getattr(self, "w_low", 1.0),
            w_high=getattr(self, "w_high", 1.0),
            fdg_levels=getattr(self, "fdg_levels", 3),
            s2_guidance_enabled=getattr(self, "s2_guidance_enabled", False),
            s2_scale_omega=getattr(self, "s2_scale_omega", 0.25),
            s2_drop_ratio=getattr(self, "s2_drop_ratio", 0.1),
        )

        # append Mahiro post‑CFG if enabled
        if getattr(self, "mahiro_enabled", False):
            import torch, torch.nn.functional as F
            def _mahiro_post(args):
                scale = float(args.get("cond_scale", 1.0))
                C = args["cond_denoised"]
                U = args["uncond_denoised"]
                cfg = args["denoised"]
                leap = C * scale
                merge = 0.5 * (leap + cfg)
                def srs(x): return torch.sqrt(x.abs() + 1e-12) * x.sign()
                u_leap = U * scale
                sim = F.cosine_similarity(srs(u_leap), srs(merge), dim=list(range(1, u_leap.ndim))).mean()
                gate = 2.0 * (sim + 1.0)
                return (gate * cfg + (4.0 - gate) * leap) / 4.0
            try:
                patched_unet.set_model_sampler_post_cfg_function(_mahiro_post, "mahiro_gate")
            except Exception:
                from ldm_patched.modules.model_patcher import set_model_options_post_cfg_function
                mo = getattr(patched_unet, "model_options", {})
                mo = set_model_options_post_cfg_function(mo, _mahiro_post, disable_cfg1_optimization=True)
                patched_unet.set_model_options(mo)

        # write back
        if hasattr(p.sd_model, 'forge_objects'):
            p.sd_model.forge_objects.unet = patched_unet
        else:
            p.sd_model = patched_unet

        # metadata
        if getattr(self, "mahiro_enabled", False):
            p.extra_generation_params["Mahiro Enabled"] = True
        p.extra_generation_params["CFG-Zero Enabled"] = bool(getattr(self, "cfg_zero_enabled", False))
        p.extra_generation_params["FDG Enabled"] = bool(getattr(self, "fdg_enabled", False))
        p.extra_generation_params["S2-Guidance Enabled"] = bool(getattr(self, "s2_guidance_enabled", False))

    except Exception as e:
        logging.exception(f"[CFG-Zero Script] process() failed: {e}")

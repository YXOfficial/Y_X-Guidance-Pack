
import logging
import gradio as gr
from modules import scripts
from guidance_pack.cfgzero.nodes_cfg_zero import CFGZeroNode
from guidance_pack.mahiro.nodes_mahiro import MahiroPack

class GuidancePackScript(scripts.Script):
    def __init__(self):
        self.state = {}

    sorting_priority = 15.05
    def title(self): return "Guidance Pack (Mahiro / CFG-Zero / FDG / S² / CFG++)"

    def show(self, is_img2img): return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Gói guidance hợp nhất cho reForge/SD-WebUI.")

            with gr.Group():
                gr.Markdown("### Mahiro")
                cb_mahiro = gr.Checkbox(label="Enable Mahiro gating", value=False)

            with gr.Group():
                gr.Markdown("### CFG-Zero / FDG / S²")
                cb_cfgzero = gr.Checkbox(label="Enable CFG-Zero", value=False)
                cb_zero_init = gr.Checkbox(label="Zero Init First Step (Experimental)", value=False)
                cb_fdg = gr.Checkbox(label="Enable FDG", value=False)
                sl_w_low = gr.Slider(label="w_low (Low-Frequency Guidance)", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                sl_w_high = gr.Slider(label="w_high (High-Frequency Guidance)", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                sl_pyr = gr.Slider(label="FDG Pyramid Levels", minimum=1, maximum=6, step=1, value=3)
                cb_s2 = gr.Checkbox(label="Enable S²-Guidance (Experimental, slow)", value=False)
                sl_s2w = gr.Slider(label="S² Scale (ω)", minimum=0.0, maximum=2.0, step=0.05, value=0.25)
                sl_s2drop = gr.Slider(label="Block Drop Ratio", minimum=0.0, maximum=0.5, step=0.01, value=0.1)

            with gr.Group():
                gr.Markdown("### CFG++ (Samplers)")
                gr.Markdown("Để dùng CFG++ bạn cần chọn sampler CFG++ tương ứng (đã đóng gói trong pack)—tương thích với hook post-cfg.")

        return [cb_mahiro, cb_cfgzero, cb_zero_init, cb_fdg, sl_w_low, sl_w_high, sl_pyr, cb_s2, sl_s2w, sl_s2drop]

    def before_process(self, p, *vals, **kwargs):
        (mahiro, cfg_zero, zero_init, fdg, w_low, w_high, pyr, s2, s2w, s2drop) = vals
        self.state = dict(mahiro=mahiro, cfg_zero=cfg_zero, zero_init=zero_init,
                          fdg=fdg, w_low=w_low, w_high=w_high, pyr=pyr, s2=s2, s2w=s2w, s2drop=s2drop)

    def process(self, p, *args):
        # Patch model just before sampling
        m = p.sd_model
        # Apply unified node: Mahiro flag is handled inside nodes_cfg_zero
        node = CFGZeroNode()
        patched, = node.patch(m, mahiro_enabled=self.state.get("mahiro", False),
                              cfg_zero_enabled=self.state.get("cfg_zero", False),
                              zero_init_first_step=self.state.get("zero_init", False),
                              fdg_enabled=self.state.get("fdg", False),
                              w_low=self.state.get("w_low", 1.0),
                              w_high=self.state.get("w_high", 1.0),
                              fdg_levels=int(self.state.get("pyr", 3)),
                              s2_guidance_enabled=self.state.get("s2", False),
                              s2_scale_omega=self.state.get("s2w", 0.25),
                              s2_drop_ratio=self.state.get("s2drop", 0.1))
        p.sd_model = patched
        p.extra_generation_params.update({
            "guidance_pack_mahiro": self.state.get("mahiro", False),
            "guidance_pack_cfg_zero": self.state.get("cfg_zero", False),
            "guidance_pack_fdg": self.state.get("fdg", False),
            "guidance_pack_s2": self.state.get("s2", False),
        })

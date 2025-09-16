
import logging
import gradio as gr
from modules import scripts
from guidance_pack.cfgzero.nodes_cfg_zero import CFGZeroNode
from guidance_pack.mahiro.append_mahiro import AppendMahiroPostCFG

class GuidancePackScript(scripts.Script):
    def __init__(self):
        self.state = {}

    sorting_priority = 15.05
    def title(self):
        return "Guidance Pack (Mahiro / CFG-Zero / FDG / S² / CFG++)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown("Gói guidance hợp nhất cho Test‑reForge / SD WebUI.")

            with gr.Group():
                gr.Markdown("### Mahiro")
                cb_mahiro = gr.Checkbox(label="Enable Mahiro gating", value=False)

            with gr.Group():
                gr.Markdown("### CFG‑Zero / FDG / S²")
                cb_cfgzero = gr.Checkbox(label="Enable CFG‑Zero", value=False)
                cb_zero_init = gr.Checkbox(label="Zero‑init first step", value=False)
                cb_fdg = gr.Checkbox(label="Enable FDG (frequency‑domain guidance)", value=False)
                sl_w_low = gr.Slider(label="w_low", minimum=0.0, maximum=10.0, step=0.1, value=1.0)
                sl_w_high = gr.Slider(label="w_high", minimum=0.0, maximum=10.0, step=0.1, value=1.0)
                sl_pyr = gr.Slider(label="FDG pyramid levels", minimum=2, maximum=8, step=1, value=3)
                cb_s2 = gr.Checkbox(label="Enable S²‑Guidance (experimental)", value=False)
                sl_s2w = gr.Slider(label="S² scale ω", minimum=0.0, maximum=2.0, step=0.05, value=0.25)
                sl_s2drop = gr.Slider(label="S² block drop ratio", minimum=0.0, maximum=0.5, step=0.01, value=0.10)

            with gr.Group():
                gr.Markdown("### CFG++ (samplers)")
                gr.Markdown("Các sampler CFG++ đã nằm trong pack (cần đăng ký vào sampler registry nếu muốn chọn từ dropdown).")

        return [cb_mahiro, cb_cfgzero, cb_zero_init, cb_fdg, sl_w_low, sl_w_high, sl_pyr, cb_s2, sl_s2w, sl_s2drop]

    def before_process(self, p, *vals, **kwargs):
        (mahiro, cfg_zero, zero_init, fdg, w_low, w_high, pyr, s2, s2w, s2drop) = vals
        self.state = dict(
            mahiro=mahiro,
            cfg_zero=cfg_zero,
            zero_init=zero_init,
            fdg=fdg,
            w_low=w_low,
            w_high=w_high,
            pyr=int(pyr),
            s2=s2,
            s2w=s2w,
            s2drop=s2drop
        )

    def process(self, p, *args):
        try:
            # Patch the UNet (ModelPatcher) not the DiffusionEngine
            unet = p.sd_model.forge_objects.unet

            # First: apply CFG‑Zero/FDG/S² (original Y_X code)
            patched_unet, = CFGZeroNode().patch(
                unet,
                cfg_zero_enabled=self.state.get("cfg_zero", False),
                zero_init_first_step=self.state.get("zero_init", False),
                fdg_enabled=self.state.get("fdg", False),
                w_low=self.state.get("w_low", 1.0),
                w_high=self.state.get("w_high", 1.0),
                fdg_levels=self.state.get("pyr", 3),
                s2_guidance_enabled=self.state.get("s2", False),
                s2_scale_omega=self.state.get("s2w", 0.25),
                s2_drop_ratio=self.state.get("s2drop", 0.10),
            )

            # Optional: append Mahiro as an additional post‑CFG gate
            if self.state.get("mahiro", False):
                patched_unet, = AppendMahiroPostCFG().patch(patched_unet, enabled=True)

            # Write back
            p.sd_model.forge_objects.unet = patched_unet

            # Metadata
            p.extra_generation_params.update({
                "guidance_pack_mahiro": self.state.get("mahiro", False),
                "guidance_pack_cfg_zero": self.state.get("cfg_zero", False),
                "guidance_pack_fdg": self.state.get("fdg", False),
                "guidance_pack_s2": self.state.get("s2", False),
            })

        except Exception as e:
            logging.exception(f"[Guidance-Pack] Process hook failed: {e}")

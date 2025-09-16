import logging
import gradio as gr
from modules import scripts

try:
    # Forge/Test‑ReForge runtime
    from ldm_patched.modules.model_patcher import set_model_options_post_cfg_function
except Exception:
    set_model_options_post_cfg_function = None

class YX_Mahiro_Addon_Script(scripts.Script):
    """Add‑only Mahiro gating that chains AFTER the original Y_X post‑CFG.
    It does not modify or replace Y_X files; it just wraps the existing post‑CFG.
    """

    sorting_priority = 16.0  # run after Y_X script (which is usually ~15)

    def title(self): return "Y_X Add‑on: Mahiro (chain post‑CFG)"
    def show(self, is_img2img): return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            en = gr.Checkbox(label="Enable Mahiro add‑on (chain after Y_X)", value=False)
        # return one arg; we won't touch Y_X UI
        return [en]

    def process_before_every_sampling(self, p, enabled=False, **kwargs):
        self.enabled = bool(enabled)

    def process(self, p, enabled=False, *args, **kwargs):
        try:
            if not self.enabled:
                return

            # Get the UNet (ModelPatcher) from Forge/Test‑ReForge
            unet = getattr(getattr(p.sd_model, "forge_objects", p.sd_model), "unet", p.sd_model)

            # Access current model_options (do NOT overwrite; we only wrap)
            mo = getattr(unet, "model_options", {}) or {}

            # Prevent double‑append across runs
            if mo.get("_mahiro_addon_chained", False):
                return

            prev_post = mo.get("sampler_post_cfg_function", None)

            import torch
            import torch.nn.functional as F

            def _mahiro_gate(args):
                # 1) run original Y_X post‑CFG first (if any)
                den = args.get("denoised", None)
                if callable(prev_post):
                    try:
                        den = prev_post(args)
                    except Exception as e:
                        logging.warning(f"[Mahiro Add‑on] prev_post failed, falling back to args['denoised']: {e}")
                if den is None:
                    den = args["denoised"]

                # 2) Mahiro gating on top of the previous result
                scale = float(args.get("cond_scale", 1.0))
                C = args["cond_denoised"]
                U = args["uncond_denoised"]
                cfg_now = den

                leap = C * scale
                merge = 0.5 * (leap + cfg_now)

                # signed sqrt normalize; cosine along channel, then mean over HxW
                def srs(x): return torch.sqrt(x.abs() + 1e-12) * x.sign()
                sim_map = F.cosine_similarity(srs(U * scale), srs(merge), dim=1)  # (B,H,W)
                sim = sim_map.mean()  # scalar

                gate = 2.0 * (sim + 1.0)  # [0,4]
                out = (gate * cfg_now + (4.0 - gate) * leap) / 4.0
                return out

            # Chain our wrapper as the new sampler_post_cfg_function
            if set_model_options_post_cfg_function is not None:
                mo = set_model_options_post_cfg_function(mo, _mahiro_gate, disable_cfg1_optimization=True)
                mo["_mahiro_addon_chained"] = True
                unet.set_model_options(mo)
            else:
                # Fallback: try direct attribute (older forks)
                mo["sampler_post_cfg_function"] = _mahiro_gate
                mo["_mahiro_addon_chained"] = True
                try:
                    unet.set_model_options(mo)
                except Exception as e:
                    logging.error(f"[Mahiro Add‑on] Failed to set model_options: {e}")

            logging.info("[Mahiro Add‑on] Chained after Y_X post‑CFG successfully.")
        except Exception as e:
            logging.exception(f"[Mahiro Add‑on] process() failed: {e}")
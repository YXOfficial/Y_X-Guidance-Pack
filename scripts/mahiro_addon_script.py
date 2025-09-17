import logging
import gradio as gr
from modules import scripts

class YX_Mahiro_Addon_Script(scripts.Script):
    """
    Add-only Mahiro gating that chains AFTER the original Y_X post-CFG.
    Does not modify or replace Y_X files; it only wraps the existing post-CFG.
    """

    sorting_priority = 16.0  # run after Y_X script (usually ~15)

    def title(self):
        return "Y_X Add-on: Mahiro (chain post-CFG)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            en = gr.Checkbox(label="Enable Mahiro add-on (chain after Y_X)", value=False)
        # Return ONE arg; we do not touch the Y_X UI
        return [en]

    def process_before_every_sampling(self, p, enabled=False, **kwargs):
        self.enabled = bool(enabled)

    def process(self, p, enabled=False, *args, **kwargs):
        try:
            if not getattr(self, "enabled", False):
                return

            # Get UNet (ModelPatcher) from Forge/Test-ReForge
            unet = None
            try:
                if hasattr(p.sd_model, "forge_objects") and hasattr(p.sd_model.forge_objects, "unet"):
                    unet = p.sd_model.forge_objects.unet
                else:
                    unet = p.sd_model
            except Exception:
                unet = getattr(p, "sd_model", None)

            if unet is None:
                logging.warning("[Mahiro Add-on] UNet not found; skipping.")
                return

            # Access current model_options (do NOT overwrite; mutate in place)
            mo = getattr(unet, "model_options", None)
            if not isinstance(mo, dict):
                # Create dict if missing
                mo = {}
                try:
                    setattr(unet, "model_options", mo)
                except Exception:
                    pass

            # Prevent double-append across runs
            if mo.get("_mahiro_addon_chained", False):
                return

            prev_post = mo.get("sampler_post_cfg_function", None)

            import torch
            import torch.nn.functional as F

            def _mahiro_gate(args):
                # 1) run original Y_X post-CFG first (if any)
                den = args.get("denoised", None)
                if callable(prev_post):
                    try:
                        den = prev_post(args)
                    except Exception as e_prev:
                        logging.warning(f"[Mahiro Add-on] prev_post failed, using args['denoised']: {e_prev}")
                if den is None:
                    den = args["denoised"]

                # 2) Mahiro gating on top of previous result
                scale = float(args.get("cond_scale", 1.0))
                C = args["cond_denoised"]
                U = args["uncond_denoised"]
                cfg_now = den

                leap = C * scale
                merge = 0.5 * (leap + cfg_now)

                def srs(x):
                    return torch.sqrt(x.abs() + 1e-12) * x.sign()

                u_leap = U * scale
                # cosine over channel dim; mean over H,W
                sim_map = F.cosine_similarity(srs(u_leap), srs(merge), dim=1)  # (B,H,W)
                sim = sim_map.mean()
                gate = 2.0 * (sim + 1.0)  # [0,4]
                out = (gate * cfg_now + (4.0 - gate) * leap) / 4.0
                return out

            # Chain our wrapper as the new sampler_post_cfg_function (add-only)
            try:
                # 1) In-place update of model_options dict (used by k-diff samplers)
                mo["sampler_post_cfg_function"] = _mahiro_gate
                mo["_mahiro_addon_chained"] = True
            except Exception as e_mo:
                logging.warning(f"[Mahiro Add-on] Could not mutate model_options dict: {e_mo}")

            try:
                # 2) Also set attribute-level hook (used by some forks)
                if hasattr(unet, "set_model_sampler_post_cfg_function"):
                    unet.set_model_sampler_post_cfg_function(_mahiro_gate, "mahiro_addon_chain")
                else:
                    setattr(unet, "sampler_post_cfg_function", _mahiro_gate)
            except Exception as e_attr:
                logging.warning(f"[Mahiro Add-on] Could not set model-level post-CFG: {e_attr}")

            logging.info("[Mahiro Add-on] Chained post-CFG via dict+attr successfully.")

        except Exception as e:
            logging.exception(f"[Mahiro Add-on] process() failed: {e}")
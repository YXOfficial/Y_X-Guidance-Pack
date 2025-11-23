# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — original formula (minimize ||eps_text - eps_uncond||_2)
# Installed at process_before_every_sampling to attach to the cloned UNet used for this run.

import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch

def make_init_noise_pre_cfg_hook(iters=20, lr=0.05):
    import torch
    state = {"done": False}

    def _hook(model, cond, uncond, x, timestep, model_options):
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        x0 = x.detach().clone()
        # Try original AdamW-min; if no-grad, fall back to nudge-min (no grad)
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.AdamW([xg], lr=float(lr))
                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)
                    eps_text, eps_uncond = calc_cond_uncond_batch(
                        model=model, cond=cond, uncond=uncond,
                        x_in=xg, timestep=timestep, model_options=model_options
                    )
                    if not (eps_text.requires_grad or eps_uncond.requires_grad):
                        raise RuntimeError("no-grad backend")

                    diff = (eps_text - eps_uncond)
                    # per-sample L2 then mean over batch (khớp bản gốc)
                    if diff.ndim >= 2:
                        loss = diff.flatten(1).norm(p=2, dim=1).mean()
                    else:
                        loss = diff.norm(p=2)

                    loss.backward()
                    opt.step()

                x_adj = xg.detach()
                mode = "orig-min(AdamW)"
        except Exception:
            # no-grad fallback: nudge-min (giảm chuẩn Δε mà không cần autograd)
            xg = x.detach().clone()
            for _ in range(int(iters)):
                with torch.no_grad():
                    eps_text, eps_uncond = calc_cond_uncond_batch(
                        model=model, cond=cond, uncond=uncond,
                        x_in=xg, timestep=timestep, model_options=model_options
                    )
                    diff = (eps_text - eps_uncond)
                    n = diff.flatten(1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-8
                    xg = xg - float(lr) * (diff / n)
            x_adj = xg.detach()
            mode = "nudge-min(no-grad)"

        with torch.no_grad():
            rms = ((x_adj - x0).pow(2).mean()).sqrt().item()
        print(f"[InitNoise] adjusted x_T ({mode}, iters={iters}, lr={lr}, ||Δx||_rms={rms:.5f})")

        state["done"] = True
        return model, cond, uncond, x_adj, timestep, model_options

    _hook.__init_noise_tag = True
    return _hook


def _sanitize_pre_list(seq):
    """Remove None/non-callables and our old tagged hooks."""
    try:
        from collections.abc import Callable
    except Exception:
        Callable = type(lambda: None)
    out = []
    for fn in (seq or []):
        # drop None and non-callables
        if fn is None:
            continue
        if not callable(fn):
            continue
        # drop previous InitNoise hooks
        if getattr(fn, "__init_noise_tag", False):
            continue
        out.append(fn)
    return out


class ScriptInitNoise(scripts.Script):
    def title(self):
        return "[Forge/reForge] Init Noise (x_T) — original (pre-CFG once)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment (original formula)", value=False)
            iters   = gr.Slider(label="Iters", minimum=1, maximum=200, step=1, value=20)
            lr      = gr.Slider(label="AdamW LR / nudge step", minimum=1e-4, maximum=0.5, step=1e-4, value=0.05)
        return [enabled, iters, lr]

    def process_before_every_sampling(self, p, enabled, iters, lr, **kwargs):
        if not enabled:
            return
        try:
            unet = p.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access unet:", e)
            return

        # 1) Clean current list (remove None/non-callables and old InitNoise hooks)
        mo = dict(unet.model_options or {})
        pre = _sanitize_pre_list(list(mo.get("sampler_pre_cfg_function", [])))
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo

        # 2) Insert fresh hook (closure → runs once)
        hook = make_init_noise_pre_cfg_hook(int(iters), float(lr))
        mo = dict(unet.model_options or {})
        pre = _sanitize_pre_list(list(mo.get("sampler_pre_cfg_function", [])))
        pre.insert(0, hook)  # run before others
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo
        print("[InitNoise] pre-CFG hook installed at process_before_every_sampling()")

        # 3) Infotext like PAG/NAG
        try:
            if hasattr(p, "extra_generation_params") and isinstance(p.extra_generation_params, dict):
                p.extra_generation_params["InitNoise (orig) iters"] = int(iters)
                p.extra_generation_params["InitNoise (orig) LR"] = float(lr)
        except Exception:
            pass

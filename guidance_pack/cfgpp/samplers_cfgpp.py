# --- START OF FILE samplers_cfgpp.py ---
# Extracted reference CFG++ samplers from user's sampling (3).py
import torch, modules.shared, ldm_patched.modules.model_patcher
from torch import no_grad
from typing import Any, Dict

from .utils_stub import to_d, get_ancestral_step, default_noise_sampler  # lightweight stubs
@torch.no_grad()
def sample_euler_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, temp[0])
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Euler method
        x = denoised + d * sigmas[i + 1]
    return x

@torch.no_grad()
def sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    eta = modules.shared.opts.euler_ancestral_cfg_pp_eta
    s_noise = modules.shared.opts.euler_ancestral_cfg_pp_s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], temp[0])
        # Euler method
        x = denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps and CFG++."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_cfg_pp_eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_cfg_pp_s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            # r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up))) works only on non-cfgpp, weird
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2m_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()

    old_uncond_denoised = None
    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x

@torch.no_grad()
def sample_dpmpp_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++ (stochastic) with CFG++."""
    eta = modules.shared.opts.dpmpp_sde_cfg_pp_eta
    s_noise = modules.shared.opts.dpmpp_sde_cfg_pp_s_noise
    r = modules.shared.opts.dpmpp_sde_cfg_pp_r
    
    if len(sigmas) <= 1:
        return x

@torch.no_grad()
def sample_dpmpp_3m_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    """DPM-Solver++(3M) SDE."""
    eta = modules.shared.opts.dpmpp_3m_sde_cfg_pp_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_cfg_pp_s_noise if s_noise is None else s_noise

    if len(sigmas) <= 1:
        return x

# --- END OF FILE samplers_cfgpp.py ---
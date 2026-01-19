import torch
from tqdm.auto import trange
from k_diffusion.sampling import BrownianTreeNoiseSampler

@torch.no_grad()
def sample_yx_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None):
    """
    YX 4M SDE: DPM-Solver++(3M) SDE extended to 4th order (4M).
    Refactored to remove experimental momentum and use standard Brownian noise.
    """
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    seed = extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # time = sigmas[i] / sigma_max # Not needed without momentum
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            
            # Standard Exponential Integrator update for ODE part
            x_diff = (-h_eta).expm1().neg() * denoised
            x = torch.exp(-h_eta) * x + x_diff

            sde_diff = 0
            if h_3 is not None:
                # 4th Order Extrapolation (4M)
                r0 = h_1 / h
                r1 = h_2 / h
                r2 = h_3 / h
                d1_0 = (denoised   - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1_2 = (denoised_2 - denoised_3) / r2
                
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                
                sde_diff = phi_3 * d1 - phi_4 * d2
                x = x + sde_diff

            elif h_2 is not None:
                # 3rd Order (3M) warmup
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                
                sde_diff = phi_2 * d1 - phi_3 * d2
                x = x + sde_diff

            elif h_1 is not None:
                # 2nd Order (2M) warmup
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                
                sde_diff = phi_2 * d
                x = x + sde_diff

            if eta:
                # SDE Noise Injection
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                x = x + noise * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
            h_1, h_2, h_3 = h, h_1, h_2

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    return x
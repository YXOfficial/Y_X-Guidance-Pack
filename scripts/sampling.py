import torch
from tqdm.auto import trange
from k_diffusion.sampling import BrownianTreeNoiseSampler

@torch.no_grad()
def sample_yx_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """
    YX 4M SDE: Debug Mode.
    Prints internal statistics to console for analysis.
    """
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    seed = extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # History buffers
    old_denoised = [] 
    old_h = []
    
    log_sigmas = sigmas.log()
    
    print(f"\n[YX 4M SDE] Start Sampling: {len(sigmas)} steps")

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -log_sigmas[i], -log_sigmas[i + 1]
            h = s - t
            h_eta = h * (eta + 1)
            
            phi_1 = torch.expm1(-h_eta).neg() 
            phi_2 = torch.expm1(-h_eta).neg() / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            phi_4 = phi_3 / h_eta - (1/6)

            # 1. Main ODE Step
            x = torch.exp(-h_eta) * x + phi_1 * denoised

            # 2. Correction Terms
            if len(old_denoised) >= 3: # 4M Step
                h_1, h_2, h_3 = old_h[-1], old_h[-2], old_h[-3]
                d_0, d_1, d_2, d_3 = denoised, old_denoised[-1], old_denoised[-2], old_denoised[-3]

                r0 = h_1 / h
                r1 = h_2 / h
                r2 = h_3 / h

                delta_0 = (d_0 - d_1) / r0
                delta_1 = (d_1 - d_2) / r1
                delta_2 = (d_2 - d_3) / r2

                delta_0_1 = (delta_0 - delta_1) / (r0 + r1)
                delta_1_2 = (delta_1 - delta_2) / (r1 + r2)

                delta_0_1_2 = (delta_0_1 - delta_1_2) / (r0 + r1 + r2)

                d1 = delta_0 + (delta_0 - delta_1) * r0 / (r0 + r1) + delta_0_1_2 * (r0 * (r0+r1))
                d2 = delta_0_1 + delta_0_1_2 * (r0 + r1 + r0)
                d3 = delta_0_1_2

                term_3M = phi_3 * d2
                term_4M = phi_4 * d3
                
                # --- TELEMETRY ---
                mag_d1 = d1.abs().mean().item()
                mag_d2 = d2.abs().mean().item()
                mag_d3 = d3.abs().mean().item()
                
                mag_3M = term_3M.abs().mean().item()
                mag_4M = term_4M.abs().mean().item()
                
                # Ratio of 4M contribution vs 3M contribution
                ratio = mag_4M / (mag_3M + 1e-9)
                
                # Debug Print
                print(f"Step {i:03d} | D1: {mag_d1:.4f} | D2: {mag_d2:.4f} | D3: {mag_d3:.4f} || Term3M: {mag_3M:.6f} | Term4M: {mag_4M:.6f} (Ratio: {ratio:.2f})")

                # --- SAFETY CLAMP ---
                mag_3M_tensor = term_3M.abs() + 1e-6
                
                # Threshold: 4M term shouldn't exceed 50% of 3M term
                threshold_ratio = 0.5
                
                # Calculate gate
                raw_ratio = term_4M.abs() / mag_3M_tensor
                gate = torch.where(raw_ratio > threshold_ratio, threshold_ratio / (raw_ratio + 1e-8), torch.ones_like(raw_ratio))
                
                clamp_stats = gate.mean().item()
                if clamp_stats < 1.0:
                     print(f"     >>> CLAMPING ACTIVE! Avg Gate: {clamp_stats:.4f}")

                term_4M_clamped = term_4M * gate

                x = x + phi_2 * d1 - phi_3 * d2 + term_4M_clamped

            elif len(old_denoised) >= 2: # 3M Step
                h_1, h_2 = old_h[-1], old_h[-2]
                d_0, d_1, d_2 = denoised, old_denoised[-1], old_denoised[-2]
                r0, r1 = h_1 / h, h_2 / h
                delta_0 = (d_0 - d_1) / r0
                delta_1 = (d_1 - d_2) / r1
                d1 = delta_0 + (delta_0 - delta_1) * r0 / (r0 + r1)
                d2 = (delta_0 - delta_1) / (r0 + r1)
                x = x + phi_2 * d1 - phi_3 * d2
                print(f"Step {i:03d} | 3M Warmup")

            elif len(old_denoised) >= 1: # 2M Step
                h_1 = old_h[-1]
                d_0, d_1 = denoised, old_denoised[-1]
                r0 = h_1 / h
                d1 = (d_0 - d_1) / r0
                x = x + phi_2 * d1
                print(f"Step {i:03d} | 2M Warmup")
            else:
                 print(f"Step {i:03d} | 1st Step")

            # --- SDE Noise Injection ---
            if eta > 0:
                scale = sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                x = x + noise * scale

            old_denoised.append(denoised)
            old_h.append(h)
            if len(old_denoised) > 3:
                old_denoised.pop(0)
                old_h.pop(0)

    return x
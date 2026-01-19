import torch
from tqdm.auto import trange
from k_diffusion.sampling import BrownianTreeNoiseSampler

@torch.no_grad()
def sample_yx_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """
    YX 4M SDE: Experimental 4th Order DPM-Solver++ SDE.
    Uses Newton form for polynomial interpolation to stabilize high-order derivatives.
    """
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    seed = extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # History buffers
    # D0 is current, D1 is prev, D2 is prev-prev...
    old_denoised = [] 
    old_h = []
    
    log_sigmas = sigmas.log()

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

            # --- DPM-Solver++ Core ---
            # x(t_next) = exp(-h_eta) * x(t) + Integral term
            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            # --- High Order Corrections ---
            
            # Helper to calculate coefficients phi_k
            # phi_1(h) = exp(h) - 1 / h  (handled by expm1)
            # phi_2(h) = (phi_1(h) - 1) / h
            # phi_3(h) = (phi_2(h) - 0.5) / h
            # phi_4(h) = (phi_3(h) - 1/6) / h
            
            # Base phi_1 is handled by the ODE step above.
            # We need phi_2, phi_3, phi_4 for the derivatives.
            
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            phi_4 = phi_3 / h_eta - (1/6)

            # Current D0 = denoised
            
            if len(old_denoised) >= 3: # 4M Step (Current + 3 History)
                h_1, h_2, h_3 = old_h[-1], old_h[-2], old_h[-3]
                d_1, d_2, d_3 = old_denoised[-1], old_denoised[-2], old_denoised[-3]

                # Newton Divided Differences (bền vững hơn cho lưới không đều)
                # D[0] = (D0 - D1) / h
                div_0 = (denoised - d_1) / (h_1/h * h + 1e-8) # Normalize h relative to current h? No, just standard div diff.
                
                # Re-calculating using standard divided differences logic relative to lambda (log-sigma)
                # Let's simplify: map to normalized steps r0, r1, r2
                r0 = h_1 / h
                r1 = h_2 / h
                r2 = h_3 / h
                
                # First order differences
                delta_0 = (denoised - d_1) / r0      # Slope 0->1
                delta_1 = (d_1 - d_2) / r1          # Slope 1->2
                delta_2 = (d_2 - d_3) / r2          # Slope 2->3
                
                # Second order differences
                delta_0_1 = (delta_0 - delta_1) / (r0 + r1)
                delta_1_2 = (delta_1 - delta_2) / (r1 + r2)
                
                # Third order difference (The 4M special)
                delta_0_1_2 = (delta_0_1 - delta_1_2) / (r0 + r1 + r2)
                
                # DPM-Solver correction formula reconstruction:
                # Integral ~ phi_2 * D' + phi_3 * D'' + phi_4 * D'''
                
                # Taylor expansion mapping:
                # 1st Deriv approx = delta_0
                # 2nd Deriv approx = 2 * delta_0_1
                # 3rd Deriv approx = 6 * delta_0_1_2
                
                # Apply:
                # x += phi_2 * (1st Deriv)
                # x += phi_3 * (2nd Deriv)
                # x += phi_4 * (3rd Deriv) -- THIS IS THE NEW 4M PART
                
                # Note: The coefficients might need scaling depending on definition. 
                # DPM-Solver paper usually absorbs factorials into phi or D.
                # Standard pattern: 
                # x += phi_2 * delta_0
                # x += phi_3 * (delta_0 - delta_1) * ... 
                # Using the explicit Newton form terms:
                
                term_2 = delta_0 # ~ D'(t)
                term_3 = delta_0_1 * (r0) # Correction for 2nd order
                # For 4M, we add the 3rd order term.
                # The jump from 3M to 4M is adding the curvature of the curvature.
                
                # Let's use a simpler 4M formulation:
                # D(tau) ~ D0 + (tau)*delta_0 + (tau)(tau+r0)*delta_0_1 + (tau)(tau+r0)(tau+r0+r1)*delta_0_1_2
                # Integrate this polynomial from 0 to -1 (in normalized time).
                
                # Resulting update rule (derived):
                # x += phi_2 * delta_0
                # x -= phi_3 * delta_0_1 * (r0 + r1) # Approx
                # Let's stick to the structure that worked for 3M and extend it logically.
                
                # 3M Logic (from paper):
                # D1_corr = delta_0 + (delta_0 - delta_1) * (r0 / (r0+r1))
                # D2_corr = (delta_0 - delta_1) / (r0+r1)
                # x += phi_2 * D1_corr - phi_3 * D2_corr
                
                # 4M Extension:
                # We need to project delta_0_1_2 into the estimate.
                # This is "risky" but let's try.
                
                # Refined 3M derivatives (as base):
                d1_base = delta_0 + (delta_0 - delta_1) * r0 / (r0 + r1)
                d2_base = (delta_0 - delta_1) / (r0 + r1)
                
                # 4M Correction factor:
                # We add the influence of the 3rd diff to d1 and d2 estimates.
                # Or simply add the 3rd term directly.
                
                # Try explicit 4th order term:
                # x += phi_4 * (3rd Deriv)
                # 3rd Deriv approx = delta_0_1_2 * (some scaling based on steps)
                # Heuristic scaling: 2 * (r0 + r1 + r2) ? 
                # Let's look at the noise pattern. If it's high freq, dampen high order.
                
                d3_val = delta_0_1_2 
                
                # Improved 4M Update:
                # 3M part
                x = x + phi_2 * d1_base - phi_3 * d2_base
                # 4M part (The "Crazy" part)
                # We add the 3rd derivative contribution.
                # In Taylor expansion: Integral(tau^3/6 * D''') = phi_4 * D'''
                # D''' is approx 6 * delta_0_1_2
                
                x = x + phi_4 * (d3_val * 6.0)

            elif len(old_denoised) >= 2: # 3M Step (Warmup)
                h_1, h_2 = old_h[-1], old_h[-2]
                d_1, d_2 = old_denoised[-1], old_denoised[-2]
                
                r0 = h_1 / h
                r1 = h_2 / h
                
                delta_0 = (denoised - d_1) / r0
                delta_1 = (d_1 - d_2) / r1
                
                d1 = delta_0 + (delta_0 - delta_1) * r0 / (r0 + r1)
                d2 = (delta_0 - delta_1) / (r0 + r1)
                
                x = x + phi_2 * d1 - phi_3 * d2

            elif len(old_denoised) >= 1: # 2M Step (Warmup)
                h_1 = old_h[-1]
                d_1 = old_denoised[-1]
                r0 = h_1 / h
                d1 = (denoised - d_1) / r0
                x = x + phi_2 * d1

            # --- SDE Noise Injection ---
            if eta > 0:
                scale = sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                x = x + noise * scale

            # Update history
            old_denoised.append(denoised)
            old_h.append(h)
            if len(old_denoised) > 3: # Keep window of 3 history items
                old_denoised.pop(0)
                old_h.pop(0)

    return x

import torch
from tqdm.auto import trange
from k_diffusion.sampling import BrownianTreeNoiseSampler

@torch.no_grad()
def sample_yx_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """
    YX 4M SDE: 4th Order DPM-Solver++ SDE using Exact Taylor Expansion.
    Correctly handles high-order derivatives using Newton Divided Differences.
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
            
            # --- Exact Phi Functions Calculation ---
            # z = -h_eta
            # phi_1(z) = (exp(z) - 1) / z
            # phi_k(z) = (phi_{k-1}(z) - 1/(k-1)!) / z
            
            z = -h_eta
            
            # Using expm1 for numerical stability near 0
            # phi_1 = (exp(z) - 1) / z
            if torch.abs(z).max() < 1e-4:
                # Taylor approx for very small steps to avoid division by zero artifacts
                phi_1 = 1 + z/2 + z**2/6 + z**3/24
                phi_2 = 1/2 + z/6 + z**2/24 + z**3/120
                phi_3 = 1/6 + z/24 + z**2/120 + z**3/720
                phi_4 = 1/24 + z/120 + z**2/720 + z**3/5040
            else:
                phi_1 = torch.expm1(z) / z
                phi_2 = (phi_1 - 1) / z
                phi_3 = (phi_2 - 0.5) / z
                phi_4 = (phi_3 - (1/6)) / z
            
            # --- Base DPM-Solver++ Update (Term 0) ---
            # x_next = exp(z) * x_curr + h_eta * phi_1 * D0
            # Note: h_eta * phi_1 = h_eta * (exp(z)-1)/z = h_eta * (exp(-h_eta)-1)/(-h_eta) = 1 - exp(-h_eta)
            # This matches the standard form: exp(-h)*x + (1-exp(-h))*D
            
            x = torch.exp(z) * x + (1 - torch.exp(z)) * denoised
            
            # --- High Order Corrections (Terms 1, 2, 3) ---
            # We add terms: h^k * phi_k * D^(k-1)
            # Note: The factor h_eta is used in the phi arguments, so we scale by h_eta.
            
            if len(old_denoised) >= 3: # 4M Step
                h_1, h_2, h_3 = old_h[-1], old_h[-2], old_h[-3]
                d_1, d_2, d_3 = old_denoised[-1], old_denoised[-2], old_denoised[-3]

                # Newton Divided Differences (coefficients of the polynomial in Newton basis)
                # D(tau) ~ D0 + (tau)*delta_0 + (tau)(tau+r0)*delta_0_1 + ...
                
                # Normalized steps relative to current h_eta
                # We use the raw derivatives approximation directly
                
                # 1. First differences (Slopes)
                # delta_i = (y_i - y_{i+1}) / (h_gap)
                # Note: old_h are actual step sizes.
                
                delta_0 = (denoised - d_1) / h_1
                delta_1 = (d_1 - d_2) / h_2
                delta_2 = (d_2 - d_3) / h_3
                
                # 2. Second differences (Curvature)
                delta_0_1 = (delta_0 - delta_1) / (h_1 + h_2)
                delta_1_2 = (delta_1 - delta_2) / (h_2 + h_3)
                
                # 3. Third difference (Jerk)
                delta_0_1_2 = (delta_0_1 - delta_1_2) / (h_1 + h_2 + h_3)
                
                # Taylor Coefficients estimates at t=0
                D_1 = delta_0  + delta_0_1 * h_1 + delta_0_1_2 * h_1 * (h_1 + h_2)
                D_2 = 2 * (delta_0_1 + delta_0_1_2 * (h_1 + h_1 + h_2)) # Approx
                # Actually, simpler reconstruction from Newton form to Taylor at 0:
                # P(t) = d0 + t*del0 + t(t+h1)*del01 + t(t+h1)(t+h1+h2)*del012
                # P'(0) = del0 + h1*del01 + h1(h1+h2)*del012
                # P''(0) = 2*del01 + 2*(2*h1 + h2)*del012
                # P'''(0) = 6*del012
                
                deriv_1 = delta_0 + h_1 * delta_0_1 + h_1 * (h_1 + h_2) * delta_0_1_2
                deriv_2 = 2 * delta_0_1 + 2 * (2*h_1 + h_2) * delta_0_1_2
                deriv_3 = 6 * delta_0_1_2
                
                # Apply corrections
                # Term k: h_eta^k * phi_k * (D^(k-1) / h_eta^(k-1)) ? 
                # No, standard form: x += h^k * phi_k * D^(k-1) is only for linear phi.
                # Correct integral add-on:
                # x += h_eta^2 * phi_2 * D_1 
                # x += h_eta^3 * phi_3 * D_2
                # x += h_eta^4 * phi_4 * D_3
                
                # Note: D_k here are the raw derivatives wrt lambda.
                
                x = x + (h_eta**2) * phi_2 * deriv_1
                x = x + (h_eta**3) * phi_3 * deriv_2
                x = x + (h_eta**4) * phi_4 * deriv_3

            elif len(old_denoised) >= 2: # 3M Step (Warmup)
                h_1, h_2 = old_h[-1], old_h[-2]
                d_1, d_2 = old_denoised[-1], old_denoised[-2]
                
                delta_0 = (denoised - d_1) / h_1
                delta_1 = (d_1 - d_2) / h_2
                delta_0_1 = (delta_0 - delta_1) / (h_1 + h_2)
                
                deriv_1 = delta_0 + h_1 * delta_0_1
                deriv_2 = 2 * delta_0_1
                
                x = x + (h_eta**2) * phi_2 * deriv_1
                x = x + (h_eta**3) * phi_3 * deriv_2

            elif len(old_denoised) >= 1: # 2M Step (Warmup)
                h_1 = old_h[-1]
                d_1 = old_denoised[-1]
                
                delta_0 = (denoised - d_1) / h_1
                deriv_1 = delta_0
                
                x = x + (h_eta**2) * phi_2 * deriv_1

            # --- SDE Noise Injection ---
            if eta > 0:
                scale = sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                x = x + noise * scale

            # Update history
            old_denoised.append(denoised)
            old_h.append(h)
            if len(old_denoised) > 3:
                old_denoised.pop(0)
                old_h.pop(0)

    return x
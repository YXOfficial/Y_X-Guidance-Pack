
# --- START OF FILE utils_stub.py ---
import torch
import modules.shared
from modules.shared import opts

def append_dims(x, target_dim):
    dims_to_add = target_dim - x.ndim
    if dims_to_add <= 0: return x
    return x[(...,) + (None,)*dims_to_add]

def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to, eta=None):
    eta = eta if eta is not None else opts.ancestral_eta
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def default_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)
# --- END OF FILE utils_stub.py ---

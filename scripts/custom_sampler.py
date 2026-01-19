import torch
import k_diffusion.sampling
from modules import sd_samplers, sd_samplers_common
try:
    from modules.sd_samplers_kdiffusion import KDiffusionSampler
except ImportError:
    # Fallback for older versions or different forks if needed, 
    # though usually this is where it resides.
    from modules.sd_samplers import KDiffusionSampler

# ------------------------------------------------------------------------------
# 1. Define your Custom Sampler Function
# ------------------------------------------------------------------------------
def sample_custom(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """
    A custom sampler implementation.
    Replace the logic below with your own sampling algorithm.
    """
    # This is a placeholder that calls Euler.
    # TODO: Implement your custom sampling loop here.
    return k_diffusion.sampling.sample_euler(
        model, x, sigmas, 
        extra_args=extra_args, 
        callback=callback, 
        disable=disable
    )

# ------------------------------------------------------------------------------
# 2. Register the Sampler with SD WebUI
# ------------------------------------------------------------------------------
def add_custom_samplers():
    # List of samplers to add
    # Format: ( "Display Name", function_reference, ["alias1", "alias2"], {options} )
    new_samplers_config = [
        ("Crazy Custom Sampler", sample_custom, ["k_crazy_custom"], {}),
    ]

    # Get existing sampler names to prevent duplicates
    if hasattr(sd_samplers, 'all_samplers'):
        existing_names = {x.name for x in sd_samplers.all_samplers}
    else:
        existing_names = set()

    samplers_data_to_add = []
    
    for label, func, aliases, options in new_samplers_config:
        if label not in existing_names:
            # Wrap the function in KDiffusionSampler so WebUI handles it correctly
            # The lambda ensures the correct function is passed to KDiffusionSampler
            data = sd_samplers_common.SamplerData(
                label,
                lambda model, funcname=func: KDiffusionSampler(funcname, model),
                aliases,
                options
            )
            samplers_data_to_add.append(data)

    if samplers_data_to_add:
        sd_samplers.all_samplers.extend(samplers_data_to_add)
        # Re-map the samplers so they are accessible by name/label
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()
        print(f"[CrazyDiffusion] Added {len(samplers_data_to_add)} custom samplers.")

# Execute registration
try:
    add_custom_samplers()
except ImportError:
    # Safely ignore if 'modules' are not available (e.g. not running in WebUI)
    pass
except Exception as e:
    print(f"[CrazyDiffusion] Error adding custom samplers: {e}")

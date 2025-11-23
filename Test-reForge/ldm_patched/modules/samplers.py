# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official


from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.unipc import uni_pc
import torch
import collections
from ldm_patched.modules import model_management
import math
import numpy as np
from scipy import stats

from modules import shared

def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = conds['area']
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
    if 'mask' in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert(mask.shape[1] == x_in.shape[2])
        assert(mask.shape[2] == x_in.shape[3])
        mask = mask[:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:,:,t:1+t,:] *= ((1.0/rr) * (t + 1))
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:,:,area[0] - 1 - t:area[0] - t,:] *= ((1.0/rr) * (t + 1))
        if area[3] != 0:
            for t in range(rr):
                mult[:,:,:,t:1+t] *= ((1.0/rr) * (t + 1))
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:,:,:,area[1] - 1 - t:area[1] - t] *= ((1.0/rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    control = conds.get('control', None)

    patches = None
    if 'gligen' in conds:
        gligen = conds['gligen']
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches['middle_patch'] = [gligen_patch]

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches'])
    return cond_obj(input_x, mult, conditioning, area, control, patches)

def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list):

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out

def compute_cond_mark(cond_or_uncond, sigmas):
    cond_or_uncond_size = int(sigmas.shape[0])

    cond_mark = []
    for cx in cond_or_uncond:
        cond_mark += [cx] * cond_or_uncond_size

    cond_mark = torch.Tensor(cond_mark).to(sigmas)
    return cond_mark

def compute_cond_indices(cond_or_uncond, sigmas):
    cl = int(sigmas.shape[0])

    cond_indices = []
    uncond_indices = []
    for i, cx in enumerate(cond_or_uncond):
        if cx == 0:
            cond_indices += list(range(i * cl, (i + 1) * cl))
        else:
            uncond_indices += list(range(i * cl, (i + 1) * cl))

    return cond_indices, uncond_indices

def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=timestep)

        c['transformer_options'] = transformer_options

        if control is not None:
            p = control
            while p is not None:
                p.transformer_options = transformer_options
                p = p.previous_controlnet
            control_cond = c.copy()  # get_control may change items in this dict, so we need to copy it
            c['control'] = control.get_control(input_x, timestep_, control_cond, len(cond_or_uncond))
            c['control_model'] = control

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond

#The main sampling function shared by all the samplers
#Returns denoised
def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    edit_strength = sum((item.get('strength', 1) for item in cond))

    skip_uncond = model_options.get("skip_uncond", False)

    if skip_uncond or (math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False)):
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

    if skip_uncond:
        cond_pred = model(x, timestep, cond=cond, model_options=model_options)
        uncond_pred = None
    else:
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred if uncond_pred is not None else None,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif skip_uncond:
        cfg_result = cond_pred
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result,
            "cond": cond,
            "uncond": uncond,
            "model": model,
            "cond_scale": cond_scale,
            "uncond_denoised": uncond_pred,
            "cond_denoised": cond_pred,
            "sigma": timestep,
            "model_options": model_options,
            "input": x
        }
        cfg_result = fn(args)

    return cfg_result

class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None):
        out = sampling_function(self.inner_model, x, timestep, uncond, cond, cond_scale, model_options=model_options, seed=seed)
        return out
    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    def __call__(self, x, sigma, cond=None, uncond=None, cond_scale=None, denoise_mask=None, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            noisy_initial_latent = self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image)
            x = x * denoise_mask + noisy_initial_latent * latent_mask
        out = self.inner_model(x, sigma, cond, uncond, cond_scale, model_options=model_options, seed=seed) if isinstance(self.inner_model, CFGNoisePredictor) else self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

def simple_scheduler(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def ddim_scheduler(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) // steps
    x = 1
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def normal_scheduler(model, steps, sgm=False, floor=False):
    s = model.model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    sgm = shared.opts.reforge_normal_sgm

    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(s.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def get_sigmas_kl_optimal(model, steps, device='cpu'):
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    
    alpha_min = torch.arctan(torch.tensor(sigma_min))
    alpha_max = torch.arctan(torch.tensor(sigma_max))
    step_indices = torch.arange(steps + 1)
    sigmas = torch.tan(step_indices / steps * alpha_min + (1.0 - step_indices / steps) * alpha_max)
    return sigmas.to(device)

def beta_scheduler(model, steps, device='cpu'):
    alpha = shared.opts.reforge_beta_dist_alpha
    beta = shared.opts.reforge_beta_dist_beta
    
    total_timesteps = len(model.model_sampling.sigmas) - 1
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)
    ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    
    sigs = [float(model.model_sampling.sigmas[int(t)]) for t in ts]
    sigs.append(0.0)
    
    return torch.FloatTensor(sigs).to(device)

def get_sigmas_karras(model, steps, device='cpu'):
    rho = shared.opts.reforge_karras_rho
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    return k_diffusion_sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)

def get_sigmas_exponential(model, steps, device='cpu'):
    shrink_factor = shared.opts.reforge_exponential_shrink_factor
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=device).exp()
    sigmas = sigmas * torch.exp(shrink_factor * torch.linspace(0, 1, steps, device=device))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_polyexponential(model, steps, device='cpu'):
    rho = shared.opts.reforge_polyexponential_rho
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    return k_diffusion_sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device)

def get_sigmas_ays_custom(model, steps, device='cpu'):
    try:
        sigmas_str = shared.opts.reforge_ays_custom_sigmas
        sigmas_values = sigmas_str.strip('[]').split(',')
        sigmas = np.array([float(x.strip()) for x in sigmas_values])
        
        if steps != len(sigmas):
            sigmas = np.interp(np.linspace(0, 1, steps), np.linspace(0, 1, len(sigmas)), sigmas)
        sigmas = np.append(sigmas, [0.0])
        return torch.FloatTensor(sigmas).to(device)
    except Exception as e:
        print(f"Error parsing custom sigmas: {e}")
        print("Falling back to default AYS sigmas")
        return k_diffusion_sampling.get_sigmas_ays(steps, model.model_sampling.sigma_min, model.model_sampling.sigma_max, model.is_sdxl, device)

def cosine_scheduler(model, steps, device='cpu'):
    sf = shared.opts.reforge_cosine_sf_factor
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    sigmas = torch.zeros(steps, device=device)
    if steps == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(steps):
            p = x / (steps-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C * sf
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def cosexpblend_scheduler(model, steps, device='cpu'):
    decay = shared.opts.reforge_cosexpblend_exp_decay
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    sigmas = []
    if steps == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = decay ** (1/(steps-1))
        E = sigma_max
        for x in range(steps):
            p = x / (steps-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)

def phi_scheduler(model, steps, device='cpu'):
    power = shared.opts.reforge_phi_power
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    sigmas = torch.zeros(steps, device=device)
    if steps == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(steps):
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(steps-1))**(phi**power))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_laplace(model, steps, device='cpu'):
    mu = shared.opts.reforge_laplace_mu
    beta = shared.opts.reforge_laplace_beta
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, steps, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_karras_dynamic(model, steps, device='cpu'):
    rho = shared.opts.reforge_karras_dynamic_rho
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    ramp = torch.linspace(0, 1, steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(steps):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/steps)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_sinusoidal_sf(model, steps, device='cpu'):
    sf = shared.opts.reforge_sinusoidal_sf_factor
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    x = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(torch.pi / 2 * x)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_invcosinusoidal_sf(model, steps, device='cpu'):
    sf = shared.opts.reforge_invcosinusoidal_sf_factor
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    x = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5*(torch.cos(x * math.pi) + 1)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_react_cosinusoidal_dynsf(model, steps, device='cpu'):
    sf = shared.opts.reforge_react_cosinusoidal_dynsf_factor
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)
    x = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_min+(sigma_max-sigma_min)*(torch.cos(x*(torch.pi/2))))/sigma_max
    sigmas = sigmas**(sf*(steps*x/steps))
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

def resolve_areas_and_cond_masks(conditions, h, w, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                area = (max(1, round(area[1] * h)), max(1, round(area[2] * w)), round(area[3] * h), round(area[4] * w))
                modified['area'] = area
                c = modified
                conditions[i] = c

        if 'mask' in c:
            mask = c['mask']
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[1] != h or mask.shape[2] != w:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

            if modified.get("set_area_to_bounds", False):
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = modified

def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif 'area' not in smallest:
                            smallest = x
                        else:
                            if smallest['area'][0] * smallest['area'][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]

def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        if 'start_percent' in x:
            timestep_start = s.percent_to_sigma(x['start_percent'])
        if 'end_percent' in x:
            timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = n

def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            x['control'].pre_run(model, percent_to_timestep_function)

def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n

def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        params["width"] = params.get("width", noise.shape[3] * 8)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    return conds

class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

class UNIPC(Sampler):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        return uni_pc.sample_unipc(model_wrap, noise, latent_image, sigmas, max_denoise=self.max_denoise(model_wrap, sigmas), extra_args=extra_args, noise_mask=denoise_mask, callback=callback, disable=disable_pbar)

class UNIPCBH2(Sampler):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        return uni_pc.sample_unipc(model_wrap, noise, latent_image, sigmas, max_denoise=self.max_denoise(model_wrap, sigmas), extra_args=extra_args, noise_mask=denoise_mask, callback=callback, variant='bh2', disable=disable_pbar)

KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm"]

class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        if self.max_denoise(model_wrap, sigmas):
            noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            noise = noise * sigmas[0]

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        if latent_image is not None:
            noise += latent_image

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return KSAMPLER(sampler_function, extra_options, inpaint_options)

def wrap_model(model):
    model_denoise = CFGNoisePredictor(model)
    return model_denoise

def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)

    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    pre_run_control(model, negative + positive)

    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    samples = sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "ays", "ays_gits", "ays_11steps", "ays_32steps", "kl_optimal", "beta", "cosine", "cosexpblend", "phi", "laplace", "karras_dynamic", "sinusoidal_sf", "invcosinusoidal_sf", "react_cosinusoidal_dynsf"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

def calculate_sigmas_scheduler(model, scheduler_name, steps, is_sdxl=False):
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)

    if scheduler_name == "karras":
        sigmas = get_sigmas_karras(model, steps)
    elif scheduler_name == "exponential":
        sigmas = get_sigmas_exponential(model, steps)
    elif scheduler_name == "polyexponential":
        sigmas = get_sigmas_polyexponential(model, steps)
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model, steps)
    elif scheduler_name == "sgm_uniform":
        sigmas = normal_scheduler(model, steps, sgm=True)
    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model, steps)
    elif scheduler_name == "ddim_uniform":
        sigmas = ddim_scheduler(model, steps)
    elif scheduler_name == "kl_optimal":
        sigmas = get_sigmas_kl_optimal(model, steps)
    elif scheduler_name == "beta":
        sigmas = beta_scheduler(model, steps)
    elif scheduler_name == "cosine":
        sigmas = cosine_scheduler(model, steps)
    elif scheduler_name == "cosexpblend":
        sigmas = cosexpblend_scheduler(model, steps)
    elif scheduler_name == "phi":
        sigmas = phi_scheduler(model, steps)
    elif scheduler_name == "laplace":
        sigmas = get_sigmas_laplace(model, steps)
    elif scheduler_name == "karras_dynamic":
        sigmas = get_sigmas_karras_dynamic(model, steps)
    elif scheduler_name == "sinusoidal_sf":
        sigmas = get_sigmas_sinusoidal_sf(model, steps)
    elif scheduler_name == "invcosinusoidal_sf":
        sigmas = get_sigmas_invcosinusoidal_sf(model, steps)
    elif scheduler_name == "react_cosinusoidal_dynsf":
        sigmas = get_sigmas_react_cosinusoidal_dynsf(model, steps)
    elif scheduler_name == "ays_custom":
        sigmas = get_sigmas_ays_custom(model, steps)
    elif scheduler_name == "ays":
        sigmas = k_diffusion_sampling.get_sigmas_ays(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, is_sdxl=is_sdxl)
    elif scheduler_name == "ays_gits":
        sigmas = k_diffusion_sampling.get_sigmas_ays_gits(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, is_sdxl=is_sdxl)
    elif scheduler_name == "ays_11steps":
        sigmas = k_diffusion_sampling.get_sigmas_ays_11steps(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, is_sdxl=is_sdxl)
    elif scheduler_name == "ays_32steps":
        sigmas = k_diffusion_sampling.get_sigmas_ays_32steps(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, is_sdxl=is_sdxl)
    else:
        print("error invalid scheduler", scheduler_name)
        return None
    return sigmas

def sampler_object(name):
    if name == "uni_pc":
        sampler = UNIPC()
    elif name == "uni_pc_bh2":
        sampler = UNIPCBH2()
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = ksampler(name)
    return sampler

class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas_scheduler(self.model, self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

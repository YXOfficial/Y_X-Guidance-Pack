# --- START OF FILE cfg_zero_script.py ---
import logging
import sys
import traceback # Added for better error reporting in on_before_ui
import gradio as gr
from modules import scripts, script_callbacks
from functools import partial
from typing import Any

# We'll import the patching class from another file
from CFGZERO.nodes_cfg_zero import CFGZeroNode

class CFGZeroScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.zero_init_first_step = False # Default value for the new option
        self.cfg_zero_node_instance = CFGZeroNode() # Instantiate the node logic

    sorting_priority = 15.2 # Slightly different from Mahiro to avoid conflict if both are somehow active

    def title(self):
        return "CFG-Zero Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Toggle CFG-Zero guidance. Modifies how conditional and unconditional guidance are combined.</i></p>")
            enabled = gr.Checkbox(label="Enable CFG-Zero", value=self.enabled)
            zero_init_first_step = gr.Checkbox(label="Zero Init First Step (Experimental)", value=self.zero_init_first_step)

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled],
            outputs=None
        )
        zero_init_first_step.change(
            lambda x: self.update_zero_init(x),
            inputs=[zero_init_first_step],
            outputs=None
        )
        # Store controls for process_before_every_sampling
        self.ui_controls = [enabled, zero_init_first_step]
        return self.ui_controls

    def update_enabled(self, value):
        self.enabled = value
        logging.debug(f"CFG-Zero: Enabled toggled to: {self.enabled}")

    def update_zero_init(self, value):
        self.zero_init_first_step = value
        logging.debug(f"CFG-Zero: Zero Init First Step toggled to: {self.zero_init_first_step}")

    def process_before_every_sampling(self, p, *args, **kwargs):
        # args will contain the values from self.ui_controls in order
        if len(args) >= 2:
            self.enabled = args[0]
            self.zero_init_first_step = args[1]
        elif len(args) == 1: # backward compatibility or partial XYZ
            self.enabled = args[0]
            # self.zero_init_first_step remains its last UI-set value or default
            logging.warning("CFG-Zero: Not enough arguments for zero_init, using current value.")
        else:
            logging.warning("CFG-Zero: Not enough arguments provided to process_before_every_sampling, using current values.")

        # Handle XYZ Grid
        xyz_settings = getattr(p, "_cfg_zero_xyz", {})
        if "enabled" in xyz_settings:
            self.enabled = xyz_settings["enabled"].lower() == "true" # XYZ grid often sends strings
        if "zero_init" in xyz_settings:
            self.zero_init_first_step = xyz_settings["zero_init"].lower() == "true"

        # Always start with a fresh clone of the original unet
        # This ensures that if CFG-Zero is disabled, or if other scripts modify the unet,
        # we are starting from a known state.
        original_unet = p.sd_model.forge_objects.unet # This is already a clone if patched before
                                                      # or the original one if not.
                                                      # To be safe, we might want to access the true original,
                                                      # but Forge's system usually handles this by re-cloning.
                                                      # Let's assume p.sd_model.forge_objects.unet is the one to modify/revert.

        current_unet = p.sd_model.forge_objects.unet
        # Check if our patch is already applied, to avoid re-patching unnecessarily or to remove it.
        # This requires the patch function to mark the model, or for us to rely on disabling.
        # For simplicity, we will re-apply or reset based on 'enabled' state.
        # It's crucial that CFGZeroNode().patch() CLONES the model internally.
        
        # The most robust way is to ensure we operate on a fresh clone if enabling,
        # or restore to original if disabling.
        # Forge's UNet is often a clone. To get the *actual* base unet for cloning:
        # base_unet = p.sd_model.forge_objects.unet_for_sampler_opt_clone() # This gets the fresh unet
        # However, if other patches are applied, this might undo them.
        # The Mahiro example's approach `unet = p.sd_model.forge_objects.unet.clone()` and then
        # `p.sd_model.forge_objects.unet = unet` (if not enabled, it's a fresh clone)
        # or `p.sd_model.forge_objects.unet = patched_unet` (if enabled) is standard.

        # Ensure we have the base unet for operations
        # Note: p.sd_model.forge_objects.unet is already a working clone.
        # If we call .clone() on it, we get a clone of potentially already patched unet.
        # To be absolutely safe and ensure no double patching or interference:
        if hasattr(p, '_original_unet_before_cfg_zero'):
            p.sd_model.forge_objects.unet = p._original_unet_before_cfg_zero.clone()
        else:
            # Store the state of the unet *before* our script potentially modifies it
            # This assumes no other script with higher priority has already cloned and stored its own original
            p._original_unet_before_cfg_zero = p.sd_model.forge_objects.unet.clone()
        
        unet_to_patch = p.sd_model.forge_objects.unet # This is now a fresh clone from original or previous state

        if not self.enabled:
            # If it was previously enabled and patched by us, ensure model_sampler_post_cfg_function is cleared.
            # The simplest way is to re-assign the cloned unet_to_patch (which is from p._original_unet_before_cfg_zero)
            # Or, if the CFGZeroNode.patch method returns the original model if not enabled, that works too.
            # The Mahiro example just re-assigns a fresh clone.
            # Clearing a specific post_cfg_function:
            if hasattr(unet_to_patch, "_cfg_zero_patched"): # A marker we can set
                unet_to_patch.set_model_sampler_post_cfg_function(None, "cfg_zero_guidance") # Name it
                delattr(unet_to_patch, "_cfg_zero_patched")
            p.sd_model.forge_objects.unet = unet_to_patch # effectively p._original_unet_before_cfg_zero.clone()
            if "cfg_zero_enabled" in p.extra_generation_params:
                del p.extra_generation_params["cfg_zero_enabled"]
            if "cfg_zero_init_first_step" in p.extra_generation_params:
                del p.extra_generation_params["cfg_zero_init_first_step"]
            logging.debug(f"CFG-Zero: Disabled. UNet restored.")
            return

        logging.debug(f"CFG-Zero: Enabling with Zero Init: {self.zero_init_first_step}")
        
        # Pass the zero_init_first_step setting to the patch method
        patched_unet = self.cfg_zero_node_instance.patch(
            unet_to_patch, # Pass the current unet (which should be a fresh clone)
            zero_init_first_step=self.zero_init_first_step
        )[0] # patch returns a tuple (model,)
        
        p.sd_model.forge_objects.unet = patched_unet
        setattr(p.sd_model.forge_objects.unet, "_cfg_zero_patched", True) # Mark it

        p.extra_generation_params.update({
            "cfg_zero_enabled": True,
            "cfg_zero_init_first_step": self.zero_init_first_step,
        })
        logging.debug(f"CFG-Zero: Enabled: {self.enabled}, Zero Init: {self.zero_init_first_step}. UNet Patched.")
        return

# --- XYZ Grid Integration ---
def cfg_zero_set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_cfg_zero_xyz"):
        p._cfg_zero_xyz = {}
    # XYZ grid sends strings "True", "False". Convert to bool for our script.
    # The script's process_before_every_sampling will handle string "true"/"false"
    p._cfg_zero_xyz[field] = str(x)


def make_cfg_zero_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py") : # Support both common names
            xyz_grid = script_data.module
            break

    if xyz_grid is None:
        logging.warning("CFG-Zero: XYZ Grid script not found.")
        return

    # Check if options already exist to prevent duplicates
    if any(x.label.startswith("(CFG-Zero)") for x in xyz_grid.axis_options):
        logging.info("CFG-Zero: XYZ Grid options already registered.")
        return
        
    cfg_zero_options = [
        xyz_grid.AxisOption(
            label="(CFG-Zero) Enabled",
            type=str, # Keep as string for XYZ, parsing happens in process_before_every_sampling
            apply=partial(cfg_zero_set_value, field="enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            label="(CFG-Zero) Zero Init First Step",
            type=str, # Keep as string
            apply=partial(cfg_zero_set_value, field="zero_init"),
            choices=lambda: ["True", "False"]
        ),
    ]
    xyz_grid.axis_options.extend(cfg_zero_options)
    logging.info("CFG-Zero: XYZ Grid options successfully registered.")


def on_cfg_zero_before_ui():
    try:
        make_cfg_zero_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] CFG-Zero Script: Error setting up XYZ Grid options:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_cfg_zero_before_ui)
# --- END OF FILE cfg_zero_script.py ---

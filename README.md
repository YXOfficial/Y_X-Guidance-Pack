# CFG-Zero for Stable Diffusion WebUI (Forge)

This is an extension for the [Stable Diffusion WebUI (Forge)](https://github.com/lllyasviel/stable-diffusion-webui-forge) that implements a version of CFG-Zero guidance.

**This is a quick patch and adaptation based on the concepts presented in the research paper and repository: [CFG-Zero*: A new guidance algorithm for diffusion models (CFG-Zero\*)](https://github.com/WeichenFan/CFG-Zero-star).**

It allows for an alternative way to combine conditional (positive prompt) and unconditional (negative prompt) guidance during the image generation process.

## Features

*   **CFG-Zero Guidance:** Implements the core logic of CFG-Zero, which aims to optimize the scaling factor (`st_star`) between conditional and unconditional predictions.
*   **Zero Init First Step (Experimental):** An optional feature to initialize the unconditional prediction to zero for the very first sampling step. This can sometimes influence the initial direction of the generation.
*   **XYZ Grid Integration:** Allows for easy comparison of CFG-Zero enabled/disabled and Zero Init First Step enabled/disabled using the XYZ Plot script.

## How CFG-Zero Works (Simplified)

Standard CFG (Classifier-Free Guidance) combines the unconditional prediction (`uncond`) and conditional prediction (`cond`) like this:
`final_pred = uncond + guidance_scale * (cond - uncond)`

CFG-Zero* proposes an optimized scaling factor `st_star` for the unconditional prediction:
`st_star = (cond_flat^T * uncond_flat) / ||uncond_flat||^2`

The final prediction then becomes:
`final_pred = uncond * st_star + guidance_scale * (cond - uncond * st_star)`

This extension applies this logic via Forge's `set_model_sampler_post_cfg_function`.

## Installation

1.  Go to the `extensions` directory in your Stable Diffusion WebUI (Forge) installation.
2.  Clone this repository:
    ```bash
    git clone <URL_OF_YOUR_REPOSITORY_HERE> CFG-Zero-SdWebui
    ```
    (Replace `<URL_OF_YOUR_REPOSITORY_HERE>` with the actual URL if you host this on GitHub/GitLab etc. If distributing as a zip, instruct users to extract it into the `extensions` folder.)
3.  Restart the Stable Diffusion WebUI (Forge).

## Usage

1.  After installation, you will find a new accordion section labeled "CFG-Zero Guidance" in the txt2img and img2img tabs (usually below the main prompt areas or near the "Script" dropdown).
2.  **Enable CFG-Zero:** Check this box to activate the CFG-Zero guidance mechanism.
3.  **Zero Init First Step (Experimental):** Check this box to enable the zero initialization for the first sampling step. Use with caution and observe its effects.
4.  The extension will automatically patch the UNet before each sampling process if enabled.

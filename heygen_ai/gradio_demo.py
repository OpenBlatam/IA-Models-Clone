from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Gradio Demo for Diffusion Model Inference
User-friendly interface for text-to-image generation using Stable Diffusion pipelines.
"""


# Model loading utility
MODEL_OPTIONS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0"
}

@gr.cache()
def load_pipeline(model_name: str, use_sdxl: bool = False):
    
    """load_pipeline function."""
if use_sdxl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
            use_safetensors=True
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    if hasattr(pipe.unet, "enable_xformers_memory_efficient_attention"):
        pipe.unet.enable_xformers_memory_efficient_attention()
    return pipe

def infer(prompt, model_choice, num_inference_steps, guidance_scale) -> Any:
    use_sdxl = "XL" in model_choice
    model_id = MODEL_OPTIONS[model_choice]
    pipe = load_pipeline(model_id, use_sdxl)
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    image = result.images[0] if hasattr(result, "images") else result[0]
    return image

EXAMPLES = [
    ["A futuristic cityscape at sunset", "Stable Diffusion XL", 30, 7.5],
    ["A cat astronaut in space, digital art", "Stable Diffusion v1.5", 25, 8.0],
    ["A photorealistic portrait of a medieval queen", "Stable Diffusion XL", 40, 6.5],
    ["A fantasy landscape with dragons and castles", "Stable Diffusion v1.5", 30, 7.0]
]

description = """
# 🖼️ Diffusion Model Demo

Enter a prompt and generate an image using state-of-the-art diffusion models (Stable Diffusion v1.5 or SDXL). Adjust the number of inference steps and guidance scale for creative control.

- **Prompt:** Describe the image you want to generate.
- **Model:** Choose between SD v1.5 and SDXL (higher quality, more compute).
- **Inference Steps:** More steps = higher quality, slower.
- **Guidance Scale:** Higher = more faithful to prompt, lower = more creative.
"""

iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe your image...", lines=2),
        gr.Radio(list(MODEL_OPTIONS.keys()), label="Model", value="Stable Diffusion XL"),
        gr.Slider(10, 60, value=30, step=1, label="Inference Steps"),
        gr.Slider(4.0, 12.0, value=7.5, step=0.1, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Diffusion Model Text-to-Image Demo",
    description=description,
    examples=EXAMPLES,
    allow_flagging="never",
    cache_examples=True,
    theme="soft"
)

match __name__:
    case "__main__":
    iface.launch() 
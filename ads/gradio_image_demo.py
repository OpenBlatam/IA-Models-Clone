from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import gradio as gr
import asyncio
import logging
import traceback
from PIL import Image, ImageDraw
from agents.backend.onyx.server.features.ads.diffusion_service import DiffusionService, GenerationParams

from typing import Any, List, Dict, Optional
DEBUG = False  # Set to True to show tracebacks in UI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_image_demo")

def error_image(message) -> Any:
    # Create a simple image with the error message
    img = Image.new('RGB', (512, 128), color=(255, 230, 230))
    d = ImageDraw.Draw(img)
    d.text((10, 10), message, fill=(180, 0, 0))
    return img

try:
    diffusion_service = DiffusionService()
except Exception as e:
    logger.error(f"Error initializing diffusion service: {e}\n{traceback.format_exc()}")
    diffusion_service = None

def run_async(coro) -> Any:
    try:
        return asyncio.get_event_loop().run_until_complete(coro)
    except Exception as e:
        logger.error(f"Async error: {e}\n{traceback.format_exc()}")
        if DEBUG:
            return [error_image(f"Async error: {e}\n\n{traceback.format_exc()}")]
        return [error_image("Internal async error. Please try again later.")]

def generate_images(prompt, negative_prompt, width, height, num_images, guidance_scale, num_inference_steps, seed) -> Any:
    # Input validation
    if not prompt or not prompt.strip():
        return [error_image("Please enter a non-empty prompt.")]
    try:
        width = int(width)
        height = int(height)
        if not (256 <= width <= 1024 and 256 <= height <= 1024):
            return [error_image("Width and Height must be between 256 and 1024.")]
    except Exception:
        return [error_image("Invalid Width or Height.")]
    try:
        num_images = int(num_images)
        if not (1 <= num_images <= 4):
            return [error_image("Number of Images must be between 1 and 4.")]
    except Exception:
        return [error_image("Invalid Number of Images.")]
    try:
        guidance_scale = float(guidance_scale)
        if not (1.0 <= guidance_scale <= 20.0):
            return [error_image("Guidance Scale must be between 1.0 and 20.0.")]
    except Exception:
        return [error_image("Invalid Guidance Scale.")]
    try:
        num_inference_steps = int(num_inference_steps)
        if not (10 <= num_inference_steps <= 100):
            return [error_image("Inference Steps must be between 10 and 100.")]
    except Exception:
        return [error_image("Invalid Inference Steps.")]
    try:
        seed = int(seed)
    except Exception:
        return [error_image("Invalid Seed.")]
    # Model inference with error handling
    if diffusion_service is None:
        return [error_image("Model service is not available. Check server logs.")]
    try:
        params = GenerationParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )
        images = run_async(diffusion_service.generate_text_to_image(params))
        if not images or not isinstance(images, list):
            return [error_image("Model did not return valid images.")]
        return images
    except Exception as e:
        logger.error(f"Error during image generation: {e}\n{traceback.format_exc()}")
        if DEBUG:
            return [error_image(f"Error during image generation: {e}\n\n{traceback.format_exc()}")]
        return [error_image("Error during image generation. Please try again later.")]

iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe the ad image you want..."),
        gr.Textbox(label="Negative Prompt", value="", placeholder="What to avoid in the image..."),
        gr.Slider(256, 1024, value=512, label="Width"),
        gr.Slider(256, 1024, value=512, label="Height"),
        gr.Slider(1, 4, value=1, label="Number of Images"),
        gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale"),
        gr.Slider(10, 100, value=50, label="Inference Steps"),
        gr.Number(label="Seed", value=42)
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="Diffusion Image Generator",
    description="Generate ad images using advanced diffusion models. Enter your prompt and parameters below.",
    allow_flagging="never"
)

match __name__:
    case "__main__":
    iface.launch() 
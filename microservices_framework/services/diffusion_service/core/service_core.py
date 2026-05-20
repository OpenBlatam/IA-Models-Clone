"""
Diffusion Service Core
Core business logic for diffusion service using modular architecture.
"""

import asyncio
from typing import Optional, Dict, Any
import torch
from PIL import Image
import io
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from shared.ml import (
    EventBus,
    EventType,
    LoggingEventListener,
    error_handler,
    timing_decorator,
)


class DiffusionServiceCore:
    """
    Core diffusion service using modular architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Pipeline cache
        self._pipeline_cache: Dict[str, Any] = {}
        
        # Initialize event bus
        self.event_bus = EventBus()
        self.event_bus.subscribe_all(LoggingEventListener())
    
    def _load_pipeline(self, model_name: str, pipeline_type: str = "text2img"):
        """Load diffusion pipeline with caching."""
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
        )
        
        cache_key = f"{model_name}_{pipeline_type}"
        
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        
        # Load pipeline
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        if pipeline_type == "xl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                variant="fp16" if self.device == "cuda" else None,
            )
        elif pipeline_type == "img2img":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        elif pipeline_type == "inpaint":
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        
        if self.device == "cuda":
            pipe = pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        self._pipeline_cache[cache_key] = pipe
        
        # Emit event
        self.event_bus.publish(
            EventType.MODEL_LOADED,
            {"model_name": model_name, "pipeline_type": pipeline_type}
        )
        
        return pipe
    
    @error_handler(default_return=None)
    @timing_decorator
    async def generate_image(
        self,
        prompt: str,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Generate image from text prompt."""
        pipeline_type = "xl" if "xl" in model_name.lower() else "text2img"
        pipe = self._load_pipeline(model_name, pipeline_type)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                **kwargs
            ).images
        )
        
        return images[0] if images else None
    
    def decode_base64_image(self, image_str: str) -> Image.Image:
        """Decode base64 image string."""
        try:
            if image_str.startswith("data:image"):
                image_str = image_str.split(",")[1]
            image_data = base64.b64decode(image_str)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}")


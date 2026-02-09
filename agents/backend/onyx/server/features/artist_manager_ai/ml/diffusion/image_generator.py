"""
Image Generator using Diffusion Models
======================================

Image generation using Stable Diffusion models.
"""

import torch
import logging
from typing import Optional, List, Dict, Any
from PIL import Image
import io

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers not available. Install with: pip install diffusers")


class ImageGenerator:
    """
    Image generator using diffusion models.
    
    Supports:
    - Stable Diffusion
    - Stable Diffusion XL
    - Custom schedulers
    - Image-to-image
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        device: Optional[str] = None,
        dtype: str = "float16"
    ):
        """
        Initialize image generator.
        
        Args:
            model_name: HuggingFace model name
            use_xl: Whether to use SDXL
            device: Device ("cpu", "cuda", or None for auto)
            dtype: Data type ("float16", "float32")
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers not installed. Install with: pip install diffusers")
        
        self.model_name = model_name
        self.use_xl = use_xl
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        
        logger.info(f"Loading diffusion model: {model_name} on {self.device}")
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load diffusion pipeline."""
        try:
            if self.use_xl:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype
                )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimize for inference
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Generate images from text prompt.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_images: Number of images to generate
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            seed: Random seed
            scheduler: Scheduler name (optional)
        
        Returns:
            List of generated images
        """
        # Set scheduler if specified
        if scheduler:
            self._set_scheduler(scheduler)
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate
        try:
            outputs = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            )
            
            return outputs.images
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def _set_scheduler(self, scheduler_name: str):
        """Set scheduler."""
        schedulers = {
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "pndm": PNDMScheduler
        }
        
        scheduler_class = schedulers.get(scheduler_name.lower())
        if scheduler_class:
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"Scheduler set to: {scheduler_name}")
    
    def generate_for_event(
        self,
        event_type: str,
        event_description: str,
        style: str = "professional"
    ) -> List[Image.Image]:
        """
        Generate image for event.
        
        Args:
            event_type: Type of event
            event_description: Event description
            style: Image style
        
        Returns:
            Generated images
        """
        prompt = f"{style} image of {event_type}: {event_description}, high quality, detailed"
        negative_prompt = "blurry, low quality, distorted"
        
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=1,
            guidance_scale=7.5
        )





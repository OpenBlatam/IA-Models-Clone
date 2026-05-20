"""
Diffusion Model Service for Image Generation

Implements Stable Diffusion pipelines for generating images
from text prompts, following best practices for diffusion models.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import torch
from PIL import Image
import numpy as np

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler
)
from diffusers.utils import export_to_video

from .base_service import BaseAIService
from ...config import settings

logger = logging.getLogger(__name__)


class DiffusionService(BaseAIService):
    """
    Service for image generation using diffusion models
    
    Supports:
    - Stable Diffusion v1.5
    - Stable Diffusion v2.1
    - Stable Diffusion XL
    """
    
    def __init__(self):
        """
        Initialize diffusion service
        
        Uses Stable Diffusion pipeline for text-to-image generation.
        """
        super().__init__(
            model_name=settings.diffusion_model,
            model_type="diffusion"
        )
        self.scheduler_type = "DPMSolverMultistepScheduler"
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.image_size = 512
        self._load_model()
    
    def _load_model_impl(self) -> None:
        """Load the diffusion pipeline"""
        if not settings.diffusion_enabled:
            logger.warning("Diffusion models are disabled")
            return
        
        try:
            logger.info(f"Loading diffusion model: {self.model_name} on {self.device}")
            
            # Determine pipeline type based on model name
            if "xl" in self.model_name.lower():
                self.model = StableDiffusionXLPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_mixed_precision and self.device.type == "cuda" else torch.float32,
                    cache_dir=settings.model_cache_dir
                )
            else:
                self.model = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_mixed_precision and self.device.type == "cuda" else torch.float32,
                    cache_dir=settings.model_cache_dir
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Setup scheduler
            self._setup_scheduler()
            
            # Enable attention slicing for memory efficiency
            if hasattr(self.model, "enable_attention_slicing"):
                self.model.enable_attention_slicing()
            
            # Enable VAE slicing for memory efficiency
            if hasattr(self.model, "enable_vae_slicing"):
                self.model.enable_vae_slicing()
            
            logger.info("Diffusion model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading diffusion model: {e}", exc_info=True)
            raise
    
    def _setup_scheduler(self) -> None:
        """Setup the noise scheduler"""
        scheduler_map = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler
        }
        
        scheduler_class = scheduler_map.get(
            self.scheduler_type,
            DPMSolverMultistepScheduler
        )
        
        self.model.scheduler = scheduler_class.from_config(
            self.model.scheduler.config
        )
        logger.info(f"Scheduler set to: {self.scheduler_type}")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        num_images: int = 1
    ) -> List[Image.Image]:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text prompt describing the image
            negative_prompt: Negative prompt (what to avoid)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (higher = more adherence to prompt)
            height: Image height
            width: Image width
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            
        Returns:
            List of PIL Images
        """
        if not self.model:
            raise RuntimeError("Diffusion model not loaded")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Set defaults
            num_inference_steps = num_inference_steps or self.num_inference_steps
            guidance_scale = guidance_scale or self.guidance_scale
            height = height or self.image_size
            width = width or self.image_size
            
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate images
            with self.inference_context():
                images = self.model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images,
                    generator=generator
                ).images
            
            logger.info(f"Generated {len(images)} image(s) from prompt: {prompt[:50]}...")
            return images
            
        except Exception as e:
            logger.error(f"Error generating image: {e}", exc_info=True)
            raise
    
    def generate_image_from_chat(
        self,
        chat_title: str,
        chat_description: Optional[str] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate image based on chat content
        
        Args:
            chat_title: Chat title
            chat_description: Optional chat description
            **kwargs: Additional arguments for generate_image
            
        Returns:
            List of PIL Images
        """
        # Create prompt from chat content
        prompt_parts = [chat_title]
        if chat_description:
            prompt_parts.append(chat_description)
        
        prompt = ", ".join(prompt_parts)
        
        return self.generate_image(prompt, **kwargs)
    
    def img2img(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> Image.Image:
        """
        Image-to-image generation (modify existing image)
        
        Args:
            prompt: Text prompt
            init_image: Initial image to modify
            strength: How much to modify (0.0 = no change, 1.0 = complete change)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            
        Returns:
            Modified PIL Image
        """
        if not self.model:
            raise RuntimeError("Diffusion model not loaded")
        
        # Check if pipeline supports img2img
        if not hasattr(self.model, "img2img"):
            raise NotImplementedError("This pipeline does not support img2img")
        
        try:
            num_inference_steps = num_inference_steps or self.num_inference_steps
            guidance_scale = guidance_scale or self.guidance_scale
            
            with self.inference_context():
                result = self.model.img2img(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in img2img: {e}", exc_info=True)
            raise
    
    def inpainting(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> Image.Image:
        """
        Inpainting (fill in masked areas of image)
        
        Args:
            prompt: Text prompt
            image: Original image
            mask_image: Mask image (white = area to fill)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            
        Returns:
            Inpainted PIL Image
        """
        if not self.model:
            raise RuntimeError("Diffusion model not loaded")
        
        # Check if pipeline supports inpainting
        if not hasattr(self.model, "inpaint"):
            raise NotImplementedError("This pipeline does not support inpainting")
        
        try:
            num_inference_steps = num_inference_steps or self.num_inference_steps
            guidance_scale = guidance_scale or self.guidance_scale
            
            with self.inference_context():
                result = self.model.inpaint(
                    prompt=prompt,
                    image=image,
                    mask_image=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in inpainting: {e}", exc_info=True)
            raise
    
    def change_scheduler(self, scheduler_type: str) -> None:
        """
        Change the noise scheduler
        
        Args:
            scheduler_type: Type of scheduler (DPMSolverMultistepScheduler, etc.)
        """
        self.scheduler_type = scheduler_type
        self._setup_scheduler()
        logger.info(f"Scheduler changed to: {scheduler_type}")
    
    def get_generation_info(self) -> Dict[str, Any]:
        """Get information about generation settings"""
        return {
            "model": self.model_name,
            "scheduler": self.scheduler_type,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_size": self.image_size,
            "device": str(self.device),
            "mixed_precision": self.use_mixed_precision
        }
















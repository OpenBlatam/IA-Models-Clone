"""
Diffusion Models - Diffusers Library Integration
================================================

Integration with HuggingFace Diffusers library for diffusion models.
"""

import torch
from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path

try:
    from diffusers import (
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        DDPMPipeline, DDIMPipeline, PNDMPipeline,
        LMSDiscreteScheduler, DPMSolverMultistepScheduler,
        UNet2DConditionModel, AutoencoderKL
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers library not available. Install with: pip install diffusers")

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from .base_model import BaseModel


class DiffusionModel(BaseModel):
    """
    Wrapper for diffusion models using Diffusers library.
    
    Supports:
    - Stable Diffusion
    - Stable Diffusion XL
    - Custom diffusion pipelines
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        model_type: str = "stable-diffusion",  # stable-diffusion, stable-diffusion-xl
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Initialize diffusion model.
        
        Args:
            model_name: Model name or path
            model_type: Type of diffusion model
            device: Target device
            torch_dtype: Model dtype
            **kwargs: Additional pipeline arguments
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required")
        
        super().__init__(device)
        
        self.model_name = model_name
        self.model_type = model_type
        
        # Determine dtype
        if torch_dtype is None:
            if self.device.type == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        try:
            if model_type == "stable-diffusion-xl":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            
            self.pipeline = self.pipeline.to(self.device)
            self._initialized = True
            
            logger.info(f"✅ Diffusion model loaded: {model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}")
            raise
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            num_images_per_prompt: Number of images per prompt
            **kwargs: Additional generation arguments
        
        Returns:
            Dictionary with generated images and metadata
        """
        try:
            with torch.no_grad():
                images = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    **kwargs
                ).images
            
            return {
                "images": images,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def save(self, path: Union[str, Path]) -> None:
        """Save pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.pipeline.save_pretrained(str(path))
        logger.info(f"✅ Diffusion model saved to {path}")


class DDPMTrainer:
    """
    Trainer for DDPM (Denoising Diffusion Probabilistic Models).
    """
    
    def __init__(
        self,
        model: UNet2DConditionModel,
        scheduler: Any,
        device: torch.device
    ):
        """
        Initialize DDPM trainer.
        
        Args:
            model: UNet model
            scheduler: Noise scheduler
            device: Target device
        """
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
    
    def train_step(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            images: Input images
            timesteps: Diffusion timesteps
            noise: Noise to predict
        
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Forward pass
        noise_pred = self.model(images, timesteps).sample
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        return {"loss": loss.item()}




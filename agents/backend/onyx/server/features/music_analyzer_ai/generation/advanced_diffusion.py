"""
Advanced Diffusion Models
Enhanced diffusion models for music generation
"""

from typing import Dict, Any, Optional, List, Union
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        DiffusionPipeline,
        StableDiffusionPipeline,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        UNet2DConditionModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class AdvancedMusicDiffusion:
    """
    Advanced diffusion model for music generation with multiple schedulers
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        scheduler_type: str = "ddim",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library required")
        
        self.model_id = model_id
        self.device = device
        self.scheduler_type = scheduler_type
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipeline = self.pipeline.to(device)
        
        # Set scheduler
        self.set_scheduler(scheduler_type)
        
        logger.info(f"Initialized Advanced Music Diffusion with {model_id}")
    
    def set_scheduler(self, scheduler_type: str):
        """Set diffusion scheduler"""
        if scheduler_type == "ddim":
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif scheduler_type == "dpm":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif scheduler_type == "euler":
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using default")
        
        self.scheduler_type = scheduler_type
    
    def generate_music_cover(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate music cover art"""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        ).images
        
        # Convert to numpy arrays
        return [np.array(img) for img in images]
    
    def generate_with_custom_scheduler(
        self,
        prompt: str,
        scheduler_type: str,
        num_inference_steps: int = 50
    ) -> List[np.ndarray]:
        """Generate with custom scheduler"""
        original_scheduler = self.scheduler_type
        self.set_scheduler(scheduler_type)
        
        try:
            images = self.generate_music_cover(
                prompt,
                num_inference_steps=num_inference_steps
            )
        finally:
            self.set_scheduler(original_scheduler)
        
        return images


class DiffusionTrainingHelper:
    """
    Helper for training diffusion models
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        noise_scheduler: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.unet = unet.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        use_amp: bool = True
    ) -> torch.Tensor:
        """Compute diffusion loss"""
        images = batch["images"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise to images
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                # Predict noise
                model_pred = self.unet(noisy_images, timesteps).sample
                loss = nn.functional.mse_loss(model_pred, noise)
        else:
            model_pred = self.unet(noisy_images, timesteps).sample
            loss = nn.functional.mse_loss(model_pred, noise)
        
        return loss
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps"""
        return torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        )


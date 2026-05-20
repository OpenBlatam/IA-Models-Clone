"""
Sampling Utilities
==================

Advanced sampling utilities for diffusion models.
"""

import torch
import numpy as np
from typing import Optional, Callable, Dict, Any
from diffusers import DDIMScheduler, DDPMScheduler
import logging

logger = logging.getLogger(__name__)


class AdvancedSampler:
    """
    Advanced sampler for diffusion models.
    
    Features:
    - Different sampling strategies
    - Custom noise schedules
    - Guidance scaling
    - Classifier-free guidance
    """
    
    def __init__(
        self,
        scheduler,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ):
        """
        Initialize advanced sampler.
        
        Args:
            scheduler: Diffusion scheduler
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
        """
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.scheduler.set_timesteps(num_inference_steps)
        self._logger = logger
    
    def sample_ddim(
        self,
        model: Callable,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        DDIM sampling.
        
        Args:
            model: Diffusion model
            latents: Initial latents
            prompt_embeds: Prompt embeddings
            guidance_scale: Guidance scale
        
        Returns:
            Generated samples
        """
        guidance_scale = guidance_scale or self.guidance_scale
        
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Predict noise
            with torch.no_grad():
                noise_pred = model(latent_model_input, t, prompt_embeds)
            
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def sample_dpm_solver(
        self,
        model: Callable,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DPM-Solver sampling (faster).
        
        Args:
            model: Diffusion model
            latents: Initial latents
            prompt_embeds: Prompt embeddings
        
        Returns:
            Generated samples
        """
        # Similar to DDIM but with DPM-Solver scheduler
        return self.sample_ddim(model, latents, prompt_embeds)
    
    def sample_with_cfg(
        self,
        model: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.
        
        Args:
            model: Diffusion model
            latents: Initial latents
            prompt_embeds: Prompt embeddings
            negative_prompt_embeds: Negative prompt embeddings
            guidance_scale: Guidance scale
        
        Returns:
            Generated samples
        """
        guidance_scale = guidance_scale or self.guidance_scale
        
        for t in self.scheduler.timesteps:
            # Concatenate for CFG
            if negative_prompt_embeds is not None:
                prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                prompt_embeds_input = torch.cat([prompt_embeds, prompt_embeds])
            
            latent_model_input = torch.cat([latents] * 2)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = model(latent_model_input, t, prompt_embeds_input)
            
            # Perform CFG
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents


class NoiseScheduler:
    """
    Custom noise scheduler utilities.
    """
    
    @staticmethod
    def linear_noise_schedule(
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ) -> torch.Tensor:
        """
        Linear noise schedule.
        
        Args:
            num_steps: Number of steps
            beta_start: Starting beta
            beta_end: Ending beta
        
        Returns:
            Beta schedule
        """
        return torch.linspace(beta_start, beta_end, num_steps)
    
    @staticmethod
    def cosine_noise_schedule(
        num_steps: int,
        s: float = 0.008
    ) -> torch.Tensor:
        """
        Cosine noise schedule.
        
        Args:
            num_steps: Number of steps
            s: Smoothing parameter
        
        Returns:
            Beta schedule
        """
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)





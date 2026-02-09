"""
Diffusion Processes Module

Implements:
- Forward diffusion process (adding noise)
- Reverse diffusion process (denoising)
- Sampling methods
"""

import logging
from typing import Optional, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)


class ForwardDiffusion:
    """Forward diffusion process: adding noise to data."""
    
    @staticmethod
    def add_noise(
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: Any
    ) -> torch.Tensor:
        """
        Add noise to clean data (forward diffusion).
        
        Implements: q(x_t | x_0) = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        
        Args:
            x_0: Clean data
            noise: Noise tensor
            timesteps: Diffusion timesteps
            scheduler: Noise scheduler
            
        Returns:
            Noisy data
        """
        # Get noise schedule values
        sqrt_alpha_cumprod = scheduler.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - scheduler.alphas_cumprod[timesteps]) ** 0.5
        
        # Expand dimensions for broadcasting
        shape = x_0.shape
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, *([1] * (len(shape) - 1)))
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(
            -1, *([1] * (len(shape) - 1))
        )
        
        # Add noise
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t
    
    @staticmethod
    def sample_noise(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Sample random noise.
        
        Args:
            shape: Shape of noise tensor
            device: Device to create noise on
            
        Returns:
            Random noise tensor
        """
        return torch.randn(shape, device=device)


class ReverseDiffusion:
    """Reverse diffusion process: denoising."""
    
    @staticmethod
    def denoise_step(
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        scheduler: Any,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion (denoising).
        
        Args:
            model_output: Model prediction (noise or velocity)
            timestep: Current timestep
            sample: Current noisy sample
            scheduler: Noise scheduler
            eta: DDIM eta parameter (0.0 = deterministic, 1.0 = stochastic)
            
        Returns:
            Denoised sample
        """
        # Use scheduler to compute previous sample
        prev_sample = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta
        ).prev_sample
        
        return prev_sample
    
    @staticmethod
    def denoise_loop(
        model: torch.nn.Module,
        scheduler: Any,
        initial_noise: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Complete denoising loop.
        
        Args:
            model: Denoising model
            scheduler: Noise scheduler
            initial_noise: Initial noise tensor
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            eta: DDIM eta parameter
            **kwargs: Additional model arguments
            
        Returns:
            Denoised sample
        """
        # Set timesteps
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        
        # Start with noise
        sample = initial_noise
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Model prediction
            model_output = model(sample, t, **kwargs)
            
            # Apply guidance if needed
            if guidance_scale > 1.0:
                # Classifier-free guidance
                uncond_output = model(sample, t, **{**kwargs, 'guidance': None})
                model_output = uncond_output + guidance_scale * (model_output - uncond_output)
            
            # Denoising step
            sample = ReverseDiffusion.denoise_step(
                model_output=model_output,
                timestep=t,
                sample=sample,
                scheduler=scheduler,
                eta=eta
            )
        
        return sample


class SamplingMethods:
    """Different sampling methods for diffusion."""
    
    @staticmethod
    def ddim_sampling(
        model: torch.nn.Module,
        scheduler: Any,
        initial_noise: torch.Tensor,
        num_inference_steps: int,
        eta: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        DDIM sampling (deterministic).
        
        Args:
            model: Denoising model
            scheduler: DDIM scheduler
            initial_noise: Initial noise
            num_inference_steps: Number of steps
            eta: DDIM eta parameter
            **kwargs: Additional arguments
            
        Returns:
            Generated sample
        """
        return ReverseDiffusion.denoise_loop(
            model=model,
            scheduler=scheduler,
            initial_noise=initial_noise,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            eta=eta,
            **kwargs
        )
    
    @staticmethod
    def ddpm_sampling(
        model: torch.nn.Module,
        scheduler: Any,
        initial_noise: torch.Tensor,
        num_inference_steps: int,
        **kwargs
    ) -> torch.Tensor:
        """
        DDPM sampling (stochastic).
        
        Args:
            model: Denoising model
            scheduler: DDPM scheduler
            initial_noise: Initial noise
            num_inference_steps: Number of steps
            **kwargs: Additional arguments
            
        Returns:
            Generated sample
        """
        return ReverseDiffusion.denoise_loop(
            model=model,
            scheduler=scheduler,
            initial_noise=initial_noise,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            eta=1.0,  # Fully stochastic
            **kwargs
        )




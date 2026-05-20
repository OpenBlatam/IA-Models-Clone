"""
Advanced Diffusion Schedulers
==============================

Advanced noise schedulers for diffusion models.
"""

import torch
import numpy as np
from typing import Optional, Union
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
import logging

logger = logging.getLogger(__name__)


class SchedulerManager:
    """
    Manager for different diffusion schedulers.
    
    Supports:
    - DDPM
    - DDIM
    - PNDM
    - DPM-Solver
    - Euler Ancestral
    """
    
    def __init__(
        self,
        scheduler_type: str = "ddpm",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        """
        Initialize scheduler manager.
        
        Args:
            scheduler_type: Type of scheduler
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
        """
        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        self.scheduler = self._create_scheduler()
        self._logger = logger
    
    def _create_scheduler(self):
        """Create scheduler based on type."""
        common_config = {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule
        }
        
        if self.scheduler_type == "ddpm":
            return DDPMScheduler(**common_config)
        elif self.scheduler_type == "ddim":
            return DDIMScheduler(**common_config)
        elif self.scheduler_type == "pndm":
            return PNDMScheduler(**common_config)
        elif self.scheduler_type == "dpm_solver":
            return DPMSolverMultistepScheduler(**common_config)
        elif self.scheduler_type == "euler_ancestral":
            return EulerAncestralDiscreteScheduler(**common_config)
        else:
            self._logger.warning(f"Unknown scheduler type: {self.scheduler_type}, using DDPM")
            return DDPMScheduler(**common_config)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples.
        
        Args:
            original_samples: Original samples
            noise: Noise tensor
            timesteps: Timesteps
        
        Returns:
            Noisy samples
        """
        return self.scheduler.add_noise(original_samples, noise, timesteps)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True
    ):
        """
        Perform one step of diffusion.
        
        Args:
            model_output: Model output
            timestep: Current timestep
            sample: Current sample
            return_dict: Whether to return dict
        
        Returns:
            Previous sample
        """
        return self.scheduler.step(model_output, timestep, sample, return_dict)
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Set inference timesteps.
        
        Args:
            num_inference_steps: Number of inference steps
        """
        self.scheduler.set_timesteps(num_inference_steps)
    
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Scale model input.
        
        Args:
            sample: Sample tensor
            timestep: Timestep
        
        Returns:
            Scaled sample
        """
        return self.scheduler.scale_model_input(sample, timestep)


class CustomScheduler:
    """
    Custom noise scheduler with configurable schedule.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule_type: str = "cosine"
    ):
        """
        Initialize custom scheduler.
        
        Args:
            num_train_timesteps: Number of timesteps
            schedule_type: Schedule type (linear, cosine, polynomial)
        """
        self.num_train_timesteps = num_train_timesteps
        self.schedule_type = schedule_type
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _compute_betas(self) -> torch.Tensor:
        """Compute beta schedule."""
        if self.schedule_type == "linear":
            return torch.linspace(0.0001, 0.02, self.num_train_timesteps)
        elif self.schedule_type == "cosine":
            s = 0.008
            steps = self.num_train_timesteps + 1
            x = torch.linspace(0, self.num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            return torch.linspace(0.0001, 0.02, self.num_train_timesteps)





"""
Diffusion Scheduler Factory Module

Factory for creating diffusion model schedulers.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        DDIMScheduler,
        DDPMScheduler,
        PNDMScheduler,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available")


class DiffusionSchedulerFactory:
    """Factory for creating diffusion schedulers."""
    
    @staticmethod
    def create(
        scheduler_type: str,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear"
    ):
        """
        Create scheduler based on type.
        
        Args:
            scheduler_type: Type of scheduler ("ddim", "ddpm", "pndm", "euler", "dpm").
            num_train_timesteps: Number of training timesteps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            beta_schedule: Beta schedule type.
        
        Returns:
            Scheduler instance.
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library required")
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "ddim":
            return DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                clip_sample=False
            )
        elif scheduler_type == "ddpm":
            return DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        elif scheduler_type == "pndm":
            return PNDMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        elif scheduler_type == "euler":
            return EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        elif scheduler_type == "dpm":
            return DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")




"""
Diffusion Schedulers Module

Provides:
- Noise scheduler implementations
- Scheduler factory
- Scheduler utilities
"""

import logging
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

# Try to import diffusers
try:
    from diffusers import (
        DDPMScheduler,
        DDIMScheduler,
        PNDMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available")


class SchedulerFactory:
    """Factory for creating noise schedulers."""
    
    @staticmethod
    def create_scheduler(
        scheduler_type: str = "ddpm",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        **kwargs
    ) -> Any:
        """
        Create noise scheduler.
        
        Args:
            scheduler_type: Type of scheduler
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
            **kwargs: Additional scheduler parameters
            
        Returns:
            Scheduler instance
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library required for schedulers")
        
        scheduler_map = {
            "ddpm": DDPMScheduler,
            "ddim": DDIMScheduler,
            "pndm": PNDMScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_ancestral": EulerAncestralDiscreteScheduler
        }
        
        if scheduler_type.lower() not in scheduler_map:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_type}. "
                f"Available: {list(scheduler_map.keys())}"
            )
        
        scheduler_class = scheduler_map[scheduler_type.lower()]
        
        # Default config
        scheduler_config = {
            "num_train_timesteps": num_train_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            **kwargs
        }
        
        scheduler = scheduler_class(**scheduler_config)
        
        logger.info(f"Created {scheduler_type} scheduler")
        
        return scheduler
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Any:
        """
        Create scheduler from configuration dictionary.
        
        Args:
            config: Scheduler configuration
            
        Returns:
            Scheduler instance
        """
        scheduler_type = config.pop("type", "ddpm")
        return SchedulerFactory.create_scheduler(
            scheduler_type=scheduler_type,
            **config
        )


def create_scheduler(
    scheduler_type: str = "ddpm",
    **kwargs
) -> Any:
    """
    Convenience function to create scheduler.
    
    Args:
        scheduler_type: Type of scheduler
        **kwargs: Scheduler parameters
        
    Returns:
        Scheduler instance
    """
    return SchedulerFactory.create_scheduler(
        scheduler_type=scheduler_type,
        **kwargs
    )




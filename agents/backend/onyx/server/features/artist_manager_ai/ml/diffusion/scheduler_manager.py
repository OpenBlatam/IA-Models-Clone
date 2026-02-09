"""
Scheduler Manager for Diffusion Models
=======================================

Manages different noise schedulers for diffusion models.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler,
        DDIMScheduler,
        LMSDiscreteScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class SchedulerManager:
    """
    Manager for diffusion model schedulers.
    
    Supports multiple scheduler types:
    - DPM Solver (fast, high quality)
    - Euler (balanced)
    - PNDM (stable)
    - DDIM (deterministic)
    - LMS (stable)
    """
    
    SCHEDULER_REGISTRY = {
        "dpm": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "pndm": PNDMScheduler,
        "ddim": DDIMScheduler,
        "lms": LMSDiscreteScheduler
    }
    
    def __init__(self):
        """Initialize scheduler manager."""
        if not DIFFUSERS_AVAILABLE:
            logger.warning("diffusers not available")
        self._logger = logger
    
    @classmethod
    def get_scheduler(cls, scheduler_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get scheduler instance.
        
        Args:
            scheduler_name: Name of scheduler
            config: Scheduler configuration
        
        Returns:
            Scheduler instance
        """
        scheduler_class = cls.SCHEDULER_REGISTRY.get(scheduler_name.lower())
        if not scheduler_class:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        if config:
            return scheduler_class.from_config(config)
        else:
            # Default config
            return scheduler_class()
    
    @staticmethod
    def list_schedulers() -> list:
        """List available schedulers."""
        return list(SchedulerManager.SCHEDULER_REGISTRY.keys())
    
    @staticmethod
    def get_recommended_steps(scheduler_name: str) -> int:
        """
        Get recommended number of steps for scheduler.
        
        Args:
            scheduler_name: Name of scheduler
        
        Returns:
            Recommended steps
        """
        recommendations = {
            "dpm": 20,
            "euler": 30,
            "pndm": 50,
            "ddim": 50,
            "lms": 50
        }
        return recommendations.get(scheduler_name.lower(), 50)





"""
Scheduler Factory
=================

Factory for creating diffusion schedulers.
"""

import logging
from typing import Optional, Type

# Third-party imports
try:
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """Factory for creating diffusion schedulers."""
    
    _scheduler_map = {
        "ddim": DDIMScheduler if DIFFUSERS_AVAILABLE else None,
        "dpm_solver": DPMSolverMultistepScheduler if DIFFUSERS_AVAILABLE else None,
    }
    
    @classmethod
    def get_scheduler(cls, scheduler_type: str) -> Optional[Type]:
        """Get scheduler class by type.
        
        Args:
            scheduler_type: Type of scheduler (ddim, dpm_solver)
        
        Returns:
            Scheduler class or None if not available
        """
        if not DIFFUSERS_AVAILABLE:
            return None
        
        return cls._scheduler_map.get(scheduler_type.lower())
    
    @classmethod
    def get_available_schedulers(cls) -> list:
        """Get list of available scheduler types.
        
        Returns:
            List of available scheduler type names
        """
        return list(cls._scheduler_map.keys())




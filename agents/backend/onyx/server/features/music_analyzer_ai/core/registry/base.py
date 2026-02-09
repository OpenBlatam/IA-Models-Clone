"""
Component Registry Base Module

Base registry class for component registration.
"""

from typing import Dict, Type, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for all components.
    Enables dynamic component discovery and registration.
    """
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._losses: Dict[str, Type] = {}
        self._optimizers: Dict[str, Callable] = {}
        self._schedulers: Dict[str, Callable] = {}
        self._metrics: Dict[str, Type] = {}
        self._callbacks: Dict[str, Type] = {}
        self._extractors: Dict[str, Callable] = {}
        self._augmentations: Dict[str, Callable] = {}
    
    def clear(self):
        """Clear all registrations."""
        self._models.clear()
        self._losses.clear()
        self._optimizers.clear()
        self._schedulers.clear()
        self._metrics.clear()
        self._callbacks.clear()
        self._extractors.clear()
        self._augmentations.clear()
        logger.info("Registry cleared")
    
    def get_all_registrations(self) -> Dict[str, Any]:
        """Get all registrations."""
        return {
            "models": list(self._models.keys()),
            "losses": list(self._losses.keys()),
            "optimizers": list(self._optimizers.keys()),
            "schedulers": list(self._schedulers.keys()),
            "metrics": list(self._metrics.keys()),
            "callbacks": list(self._callbacks.keys()),
            "extractors": list(self._extractors.keys()),
            "augmentations": list(self._augmentations.keys())
        }




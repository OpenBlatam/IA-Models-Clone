"""
Unified Factory Submodule
Aggregates factory components.
"""

from typing import Optional
from ...core.registry import get_registry
from .model import ModelFactoryMixin
from .training import TrainingFactoryMixin
from .inference import InferenceFactoryMixin
from .config import ConfigFactoryMixin


class UnifiedFactory(
    ModelFactoryMixin,
    TrainingFactoryMixin,
    InferenceFactoryMixin,
    ConfigFactoryMixin
):
    """
    Unified factory that integrates all modular components.
    Uses registry for component discovery.
    """
    
    def __init__(self):
        self.registry = get_registry()


# Global factory instance
_factory: Optional[UnifiedFactory] = None


def get_factory() -> UnifiedFactory:
    """Get global unified factory"""
    global _factory
    if _factory is None:
        _factory = UnifiedFactory()
    return _factory


__all__ = [
    "UnifiedFactory",
    "ModelFactoryMixin",
    "TrainingFactoryMixin",
    "InferenceFactoryMixin",
    "ConfigFactoryMixin",
    "get_factory",
]


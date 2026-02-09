"""
Component Registry Submodule
Aggregates registry components.
"""

from typing import Type, Callable, Optional, Dict
import logging
from .base import ComponentRegistry as BaseRegistry
from .models import ModelRegistry
from .training import TrainingComponentRegistry
from .components import ComponentRegistryMixin

logger = logging.getLogger(__name__)


class ComponentRegistry(BaseRegistry):
    """
    Complete component registry combining all functionality.
    Uses composition to combine specialized registries.
    """
    
    def __init__(self):
        # Initialize base
        super().__init__()
        
        # Initialize specialized registries
        self._model_registry = ModelRegistry()
        self._training_registry = TrainingComponentRegistry()
        self._component_registry = ComponentRegistryMixin()
        
        # Share dictionaries
        self._models = self._model_registry._models
        self._losses = self._training_registry._losses
        self._optimizers = self._training_registry._optimizers
        self._schedulers = self._training_registry._schedulers
        self._metrics = self._component_registry._metrics
        self._callbacks = self._component_registry._callbacks
        self._extractors = self._component_registry._extractors
        self._augmentations = self._component_registry._augmentations
    
    # Delegate model methods
    def register_model(self, name: str, model_class: Type):
        """Register a model class."""
        self._model_registry.register_model(name, model_class)
    
    def get_model(self, name: str) -> Optional[Type]:
        """Get model class by name."""
        return self._model_registry.get_model(name)
    
    def list_models(self) -> list:
        """List all registered models."""
        return self._model_registry.list_models()
    
    # Delegate training methods
    def register_loss(self, name: str, loss_class: Type):
        """Register a loss class."""
        self._training_registry.register_loss(name, loss_class)
    
    def get_loss(self, name: str) -> Optional[Type]:
        """Get loss class by name."""
        return self._training_registry.get_loss(name)
    
    def register_optimizer(self, name: str, optimizer_factory: Callable):
        """Register an optimizer factory."""
        self._training_registry.register_optimizer(name, optimizer_factory)
    
    def get_optimizer_factory(self, name: str) -> Optional[Callable]:
        """Get optimizer factory by name."""
        return self._training_registry.get_optimizer_factory(name)
    
    def register_scheduler(self, name: str, scheduler_factory: Callable):
        """Register a scheduler factory."""
        self._training_registry.register_scheduler(name, scheduler_factory)
    
    def get_scheduler_factory(self, name: str) -> Optional[Callable]:
        """Get scheduler factory by name."""
        return self._training_registry.get_scheduler_factory(name)
    
    # Delegate component methods
    def register_metric(self, name: str, metric_class: Type):
        """Register a metric class."""
        self._component_registry.register_metric(name, metric_class)
    
    def get_metric(self, name: str) -> Optional[Type]:
        """Get metric class by name."""
        return self._component_registry.get_metric(name)
    
    def register_callback(self, name: str, callback_class: Type):
        """Register a callback class."""
        self._component_registry.register_callback(name, callback_class)
    
    def get_callback(self, name: str) -> Optional[Type]:
        """Get callback class by name."""
        return self._component_registry.get_callback(name)
    
    def register_extractor(self, name: str, extractor_factory: Callable):
        """Register a feature extractor factory."""
        self._component_registry.register_extractor(name, extractor_factory)
    
    def get_extractor_factory(self, name: str) -> Optional[Callable]:
        """Get extractor factory by name."""
        return self._component_registry.get_extractor_factory(name)
    
    def register_augmentation(self, name: str, augmentation_factory: Callable):
        """Register a data augmentation factory."""
        self._component_registry.register_augmentation(name, augmentation_factory)
    
    def get_augmentation_factory(self, name: str) -> Optional[Callable]:
        """Get augmentation factory by name."""
        return self._component_registry.get_augmentation_factory(name)
    
    def get_all_registered(self) -> Dict[str, list]:
        """Get all registered components."""
        return {
            "models": self.list_models(),
            "losses": list(self._losses.keys()),
            "optimizers": list(self._optimizers.keys()),
            "schedulers": list(self._schedulers.keys()),
            "metrics": list(self._metrics.keys()),
            "callbacks": list(self._callbacks.keys()),
            "extractors": list(self._extractors.keys()),
            "augmentations": list(self._augmentations.keys())
        }


# Global registry instance
_registry: Optional[ComponentRegistry] = None


def get_registry() -> ComponentRegistry:
    """
    Get global component registry.
    
    Returns:
        Global ComponentRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
    return _registry


def register_model(name: str, model_class: Type):
    """Register a model in global registry."""
    get_registry().register_model(name, model_class)


def register_loss(name: str, loss_class: Type):
    """Register a loss in global registry."""
    get_registry().register_loss(name, loss_class)


def register_optimizer(name: str, optimizer_factory: Callable):
    """Register an optimizer in global registry."""
    get_registry().register_optimizer(name, optimizer_factory)


def register_scheduler(name: str, scheduler_factory: Callable):
    """Register a scheduler in global registry."""
    get_registry().register_scheduler(name, scheduler_factory)


__all__ = [
    "ComponentRegistry",
    "ModelRegistry",
    "TrainingComponentRegistry",
    "ComponentRegistryMixin",
    "get_registry",
    "register_model",
    "register_loss",
    "register_optimizer",
    "register_scheduler",
]

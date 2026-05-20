"""
Component Registry System
Centralized registry for all modular components
"""

from typing import Dict, Type, Any, Optional, Callable
import logging
from abc import ABC

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for all components
    Enables dynamic component discovery and registration
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
    
    # Model registration
    def register_model(self, name: str, model_class: Type):
        """Register a model class"""
        if name in self._models:
            logger.warning(f"Model {name} already registered, overwriting")
        self._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[Type]:
        """Get model class by name"""
        return self._models.get(name)
    
    def list_models(self) -> list:
        """List all registered models"""
        return list(self._models.keys())
    
    # Loss registration
    def register_loss(self, name: str, loss_class: Type):
        """Register a loss class"""
        if name in self._losses:
            logger.warning(f"Loss {name} already registered, overwriting")
        self._losses[name] = loss_class
        logger.info(f"Registered loss: {name}")
    
    def get_loss(self, name: str) -> Optional[Type]:
        """Get loss class by name"""
        return self._losses.get(name)
    
    # Optimizer registration
    def register_optimizer(self, name: str, optimizer_factory: Callable):
        """Register an optimizer factory"""
        if name in self._optimizers:
            logger.warning(f"Optimizer {name} already registered, overwriting")
        self._optimizers[name] = optimizer_factory
        logger.info(f"Registered optimizer: {name}")
    
    def get_optimizer_factory(self, name: str) -> Optional[Callable]:
        """Get optimizer factory by name"""
        return self._optimizers.get(name)
    
    # Scheduler registration
    def register_scheduler(self, name: str, scheduler_factory: Callable):
        """Register a scheduler factory"""
        if name in self._schedulers:
            logger.warning(f"Scheduler {name} already registered, overwriting")
        self._schedulers[name] = scheduler_factory
        logger.info(f"Registered scheduler: {name}")
    
    def get_scheduler_factory(self, name: str) -> Optional[Callable]:
        """Get scheduler factory by name"""
        return self._schedulers.get(name)
    
    # Metric registration
    def register_metric(self, name: str, metric_class: Type):
        """Register a metric class"""
        if name in self._metrics:
            logger.warning(f"Metric {name} already registered, overwriting")
        self._metrics[name] = metric_class
        logger.info(f"Registered metric: {name}")
    
    def get_metric(self, name: str) -> Optional[Type]:
        """Get metric class by name"""
        return self._metrics.get(name)
    
    # Callback registration
    def register_callback(self, name: str, callback_class: Type):
        """Register a callback class"""
        if name in self._callbacks:
            logger.warning(f"Callback {name} already registered, overwriting")
        self._callbacks[name] = callback_class
        logger.info(f"Registered callback: {name}")
    
    def get_callback(self, name: str) -> Optional[Type]:
        """Get callback class by name"""
        return self._callbacks.get(name)
    
    # Feature extractor registration
    def register_extractor(self, name: str, extractor_fn: Callable):
        """Register a feature extractor function"""
        if name in self._extractors:
            logger.warning(f"Extractor {name} already registered, overwriting")
        self._extractors[name] = extractor_fn
        logger.info(f"Registered extractor: {name}")
    
    def get_extractor(self, name: str) -> Optional[Callable]:
        """Get extractor function by name"""
        return self._extractors.get(name)
    
    # Augmentation registration
    def register_augmentation(self, name: str, aug_fn: Callable):
        """Register an augmentation function"""
        if name in self._augmentations:
            logger.warning(f"Augmentation {name} already registered, overwriting")
        self._augmentations[name] = aug_fn
        logger.info(f"Registered augmentation: {name}")
    
    def get_augmentation(self, name: str) -> Optional[Callable]:
        """Get augmentation function by name"""
        return self._augmentations.get(name)
    
    def get_all_registered(self) -> Dict[str, list]:
        """Get all registered components"""
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
    """Get global component registry"""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
    return _registry


def register_model(name: str, model_class: Type):
    """Convenience function to register model"""
    get_registry().register_model(name, model_class)


def register_loss(name: str, loss_class: Type):
    """Convenience function to register loss"""
    get_registry().register_loss(name, loss_class)


def register_optimizer(name: str, optimizer_factory: Callable):
    """Convenience function to register optimizer"""
    get_registry().register_optimizer(name, optimizer_factory)


def register_scheduler(name: str, scheduler_factory: Callable):
    """Convenience function to register scheduler"""
    get_registry().register_scheduler(name, scheduler_factory)




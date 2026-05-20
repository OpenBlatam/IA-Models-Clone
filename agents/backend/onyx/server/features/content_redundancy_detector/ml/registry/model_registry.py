"""
Model Registry
Registry pattern for model registration and retrieval
"""

import torch.nn as nn
from typing import Dict, Type, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for model classes and factories
    """
    
    def __init__(self):
        """Initialize registry"""
        self._models: Dict[str, Type[nn.Module]] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        model_class: Optional[Type[nn.Module]] = None,
        factory: Optional[Callable] = None,
    ):
        """
        Register model or factory
        
        Args:
            name: Model name
            model_class: Model class (optional)
            factory: Factory function (optional)
        """
        if model_class is not None:
            self._models[name] = model_class
            logger.info(f"Registered model class: {name}")
        
        if factory is not None:
            self._factories[name] = factory
            logger.info(f"Registered model factory: {name}")
    
    def get_model_class(self, name: str) -> Optional[Type[nn.Module]]:
        """
        Get model class by name
        
        Args:
            name: Model name
            
        Returns:
            Model class or None
        """
        return self._models.get(name)
    
    def create_model(self, name: str, **kwargs) -> Optional[nn.Module]:
        """
        Create model using factory
        
        Args:
            name: Model name
            **kwargs: Model arguments
            
        Returns:
            Model instance or None
        """
        factory = self._factories.get(name)
        if factory:
            return factory(**kwargs)
        
        model_class = self._models.get(name)
        if model_class:
            return model_class(**kwargs)
        
        logger.warning(f"Model {name} not found in registry")
        return None
    
    def list_models(self) -> list[str]:
        """List all registered models"""
        return list(set(list(self._models.keys()) + list(self._factories.keys())))


# Global registry instance
model_registry = ModelRegistry()




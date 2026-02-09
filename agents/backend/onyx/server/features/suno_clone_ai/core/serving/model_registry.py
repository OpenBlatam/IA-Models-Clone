"""
Model Registry

Registry for managing multiple models.
"""

import logging
from typing import Dict, Optional, Any
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing models."""
    
    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, nn.Module] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        model: nn.Module,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a model.
        
        Args:
            name: Model name
            model: Model instance
            metadata: Optional metadata
        """
        self.models[name] = model
        self.metadata[name] = metadata or {}
        logger.info(f"Registered model: {name}")
    
    def get(
        self,
        name: str
    ) -> Optional[nn.Module]:
        """
        Get model by name.
        
        Args:
            name: Model name
            
        Returns:
            Model instance or None
        """
        return self.models.get(name)
    
    def get_metadata(
        self,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Args:
            name: Model name
            
        Returns:
            Metadata dictionary or None
        """
        return self.metadata.get(name)
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self.models.keys())
    
    def unregister(self, name: str) -> None:
        """
        Unregister a model.
        
        Args:
            name: Model name
        """
        if name in self.models:
            del self.models[name]
            del self.metadata[name]
            logger.info(f"Unregistered model: {name}")


# Global registry instance
_global_registry = ModelRegistry()


def register_model(
    name: str,
    model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Register model in global registry."""
    _global_registry.register(name, model, metadata)


def get_model_from_registry(name: str) -> Optional[nn.Module]:
    """Get model from global registry."""
    return _global_registry.get(name)




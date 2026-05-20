"""
Model Factory
Factory pattern for creating models
"""

from typing import Optional, Dict, Any, Type
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating models
    Supports registration and creation of different model types
    """
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type):
        """
        Register a model class
        
        Args:
            name: Model name
            model_class: Model class
        """
        cls._registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create(
        cls,
        model_type: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        Create model instance
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            device: Device to use
            
        Returns:
            Model instance
        """
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._registry.keys())}")
        
        model_class = cls._registry[model_type]
        
        try:
            model = model_class(**config)
            if device:
                model = model.to(device)
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            raise
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models"""
        return list(cls._registry.keys())


class ModelBuilder:
    """
    Builder pattern for constructing models step by step
    """
    
    def __init__(self):
        """Initialize builder"""
        self.config: Dict[str, Any] = {}
        self.device: Optional[torch.device] = None
    
    def with_device(self, device: torch.device) -> 'ModelBuilder':
        """Set device"""
        self.device = device
        return self
    
    def with_config(self, **kwargs) -> 'ModelBuilder':
        """Add configuration"""
        self.config.update(kwargs)
        return self
    
    def with_mixed_precision(self, enabled: bool = True) -> 'ModelBuilder':
        """Enable mixed precision"""
        self.config['use_mixed_precision'] = enabled
        return self
    
    def build(self, model_type: str) -> nn.Module:
        """
        Build model
        
        Args:
            model_type: Type of model to build
            
        Returns:
            Built model
        """
        model = ModelFactory.create(model_type, self.config, self.device)
        return model
    
    def reset(self) -> 'ModelBuilder':
        """Reset builder"""
        self.config = {}
        self.device = None
        return self














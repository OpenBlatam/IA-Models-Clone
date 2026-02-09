"""
Model Factory Module

Model and loss creation functionality.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelFactoryMixin:
    """Model factory mixin."""
    
    def create_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ) -> nn.Module:
        """
        Create model using registry.
        
        Args:
            model_type: Model type name (must be registered)
            config: Model configuration
            device: Target device
        
        Returns:
            Model instance
        """
        config = config or {}
        
        # Get model class from registry
        model_class = self.registry.get_model(model_type)
        if model_class is None:
            raise ValueError(f"Model type '{model_type}' not found in registry. "
                           f"Available: {self.registry.list_models()}")
        
        # Create model
        model = model_class(**config)
        model = model.to(device)
        
        logger.info(f"Created {model_type} model on {device}")
        return model
    
    def create_loss(
        self,
        loss_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Create loss function.
        
        Args:
            loss_type: Loss type name
            config: Loss configuration
        
        Returns:
            Loss function
        """
        from ...training.components import (
            ClassificationLoss,
            RegressionLoss,
            MultiTaskLoss
        )
        
        config = config or {}
        
        # Try registry first
        loss_class = self.registry.get_loss(loss_type)
        if loss_class:
            return loss_class(**config)
        
        # Fallback to built-in losses
        if loss_type == "classification":
            return ClassificationLoss(**config)
        elif loss_type == "regression":
            return RegressionLoss(**config)
        elif loss_type == "multi_task":
            return MultiTaskLoss(**config)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")




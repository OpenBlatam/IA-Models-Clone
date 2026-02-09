"""
Model Creation Module

Model creation functionality.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .manager import ModelManager
from ...utils.initialization import initialize_weights


class ModelCreationMixin:
    """Model creation mixin."""
    
    def create_model(
        self: ModelManager,
        model_name: str,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        compile_model: bool = True
    ) -> nn.Module:
        """
        Create and register a model.
        
        Args:
            model_name: Name for the model.
            model_type: Type of model (must be registered).
            config: Model configuration.
            compile_model: Whether to compile model.
        
        Returns:
            Model instance.
        """
        # Create model using factory
        model = self.factory.create_model(
            model_type=model_type,
            config=config or {},
            device=self.device_manager.get_device()
        )
        
        # Initialize weights
        initialize_weights(model, strategy="xavier")
        
        # Compile if requested
        if compile_model:
            model = self.device_manager.compile_model(model)
        
        # Register model
        self.models[model_name] = model
        
        logger.info(f"Created and registered model: {model_name}")
        return model




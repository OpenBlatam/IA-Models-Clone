"""
Model Factory
=============

Factory for creating model instances.
Refactored to work with new model architecture.
"""

import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..base import BaseModel
from ..models import (
    EventDurationPredictor,
    RoutineCompletionPredictor,
    OptimalTimePredictor
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating model instances.
    
    Supports:
    - Creating models from config dict or kwargs
    - Loading pre-trained models
    - Model registry
    - Default configurations
    """
    
    # Model registry
    MODEL_REGISTRY = {
        "event_duration": EventDurationPredictor,
        "routine_completion": RoutineCompletionPredictor,
        "optimal_time": OptimalTimePredictor
    }
    
    def __init__(self):
        """Initialize factory."""
        self._logger = logger
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """
        Register a new model class.
        
        Args:
            name: Model name
            model_class: Model class (must inherit from BaseModel)
        
        Raises:
            ValueError: If model_class doesn't inherit from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(
                f"Model class must inherit from BaseModel, got {model_class}"
            )
        
        cls.MODEL_REGISTRY[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def create(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create model instance.
        
        Args:
            model_type: Type of model (must be in registry)
            config: Model configuration dictionary (optional)
            model_path: Path to pre-trained model (optional)
            **kwargs: Additional config parameters (merged with config)
        
        Returns:
            Model instance (BaseModel)
        
        Raises:
            ValueError: If model_type is not in registry
            FileNotFoundError: If model_path is provided but doesn't exist
        """
        # Validate model type
        if model_type not in self.MODEL_REGISTRY:
            available = list(self.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {available}"
            )
        
        model_class = self.MODEL_REGISTRY[model_type]
        
        # Merge config and kwargs
        if config is None:
            config = {}
        
        # Merge kwargs into config (kwargs take precedence)
        final_config = {**config, **kwargs}
        
        # Use defaults if config is empty
        if not final_config:
            final_config = self._get_default_config(model_type)
        
        # Validate model path if provided
        if model_path:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(
                    f"Model path does not exist: {model_path}"
                )
        
        # Create model (models now accept kwargs directly)
        try:
            model = model_class(**final_config)
        except Exception as e:
            self._logger.error(f"Error creating model: {str(e)}")
            raise
        
        # Load weights if path provided
        if model_path:
            try:
                model.load(str(model_path))
                self._logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                self._logger.warning(
                    f"Could not load model weights: {str(e)}"
                )
                # Continue without loading weights
        
        return model
    
    def _get_default_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for model type.
        
        Args:
            model_type: Model type
        
        Returns:
            Default configuration dictionary
        """
        defaults = {
            "event_duration": {
                "input_dim": 32,
                "hidden_dims": [128, 64, 32],
                "dropout_rate": 0.2,
                "use_batch_norm": True,
                "device": "auto",
                "dtype": "float32"
            },
            "routine_completion": {
                "input_dim": 16,
                "lstm_hidden": 64,
                "lstm_layers": 2,
                "fc_dims": [128, 64],
                "dropout_rate": 0.2,
                "device": "auto",
                "dtype": "float32"
            },
            "optimal_time": {
                "input_dim": 24,
                "hidden_dim": 128,
                "num_hours": 24,
                "num_heads": 4,
                "dropout_rate": 0.2,
                "device": "auto",
                "dtype": "float32"
            }
        }
        
        return defaults.get(model_type, {})
    
    @staticmethod
    def list_models() -> list:
        """
        List available models.
        
        Returns:
            List of model type names
        """
        return list(ModelFactory.MODEL_REGISTRY.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Model type
        
        Returns:
            Model information dictionary
        
        Raises:
            ValueError: If model_type is not in registry
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.MODEL_REGISTRY[model_type]
        default_config = self._get_default_config(model_type)
        
        return {
            "name": model_type,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "default_config": default_config
        }

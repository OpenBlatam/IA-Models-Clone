"""
Model Factory Pattern
Centralized model creation with proper configuration
"""

from typing import Dict, Any, Optional, Type
import torch.nn as nn
import logging

from models.base import BaseModel, ModelFactory, ModelConfig
from models.pytorch_models import (
    SkinAnalysisCNN,
    SkinQualityRegressor,
    ConditionClassifier,
    EnhancedSkinAnalyzer
)
from models.vision_transformers import (
    VisionTransformer,
    ViTSkinAnalyzer,
    LoRAViT
)

logger = logging.getLogger(__name__)


class SkinAnalysisModelFactory:
    """
    Factory for creating skin analysis models
    Implements factory pattern with proper configuration
    """
    
    _model_registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """Register a model class"""
        cls._model_registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create(
        cls,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create model instance
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        config = config or {}
        config.update(kwargs)
        
        if model_name in cls._model_registry:
            model_class = cls._model_registry[model_name]
            return model_class(**config)
        
        # Try ModelFactory as fallback
        try:
            return ModelFactory.create(model_name, **config)
        except ValueError:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(cls._model_registry.keys())}"
            )
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models"""
        return list(cls._model_registry.keys())


# Auto-register models
SkinAnalysisModelFactory.register_model("skin_cnn", SkinAnalysisCNN)
SkinAnalysisModelFactory.register_model("quality_regressor", SkinQualityRegressor)
SkinAnalysisModelFactory.register_model("condition_classifier", ConditionClassifier)
SkinAnalysisModelFactory.register_model("enhanced_analyzer", EnhancedSkinAnalyzer)
SkinAnalysisModelFactory.register_model("vision_transformer", VisionTransformer)
SkinAnalysisModelFactory.register_model("vit_skin", ViTSkinAnalyzer)
SkinAnalysisModelFactory.register_model("lora_vit", LoRAViT)














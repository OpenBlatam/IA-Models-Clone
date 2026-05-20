"""
Specialized Model Factory

Creates model instances following factory pattern with proper
configuration and initialization.
"""

import logging
from typing import Dict, Any, Optional, Type
import torch
import torch.nn as nn

from ..interfaces.base import IModel, IEmbeddingModel, IClassifier
from ..models.modular_transformer import (
    ModularTransformerEncoder,
    ModularMusicClassifier
)
from ..models.architectures import (
    MultiHeadAttention,
    FeedForward,
    LearnedPositionalEncoding,
    MusicFeatureEmbedding
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating model instances.
    
    Supports:
    - Transformer models
    - Classification models
    - Embedding models
    - Custom architectures
    """
    
    _model_registry: Dict[str, Type[nn.Module]] = {
        "transformer_encoder": ModularTransformerEncoder,
        "music_classifier": ModularMusicClassifier,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]):
        """
        Register a custom model class.
        
        Args:
            name: Model name
            model_class: Model class
        """
        cls._model_registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create(
        cls,
        model_type: str,
        config: Dict[str, Any],
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        if model_type not in cls._model_registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._model_registry.keys())}"
            )
        
        model_class = cls._model_registry[model_type]
        
        try:
            # Merge config and kwargs
            merged_config = {**config, **kwargs}
            
            # Create model
            model = model_class(**merged_config)
            
            logger.info(f"Created model: {model_type} with config: {config}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {e}")
            raise
    
    @classmethod
    def create_transformer_encoder(
        cls,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        **kwargs
    ) -> ModularTransformerEncoder:
        """
        Create a transformer encoder.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            ff_dim: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            Transformer encoder model
        """
        return cls.create(
            "transformer_encoder",
            {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "ff_dim": ff_dim,
                "dropout": dropout,
                "max_seq_len": max_seq_len,
                **kwargs
            }
        )
    
    @classmethod
    def create_music_classifier(
        cls,
        num_classes: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ) -> ModularMusicClassifier:
        """
        Create a music classification model.
        
        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            ff_dim: Feedforward dimension
            dropout: Dropout rate
            **kwargs: Additional arguments
            
        Returns:
            Music classifier model
        """
        return cls.create(
            "music_classifier",
            {
                "num_classes": num_classes,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "ff_dim": ff_dim,
                "dropout": dropout,
                **kwargs
            }
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """
        Create model from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and other params
            
        Returns:
            Model instance
        """
        model_type = config.pop("type", "transformer_encoder")
        return cls.create(model_type, config)


__all__ = [
    "ModelFactory",
]

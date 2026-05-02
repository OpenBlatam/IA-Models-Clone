"""
Model Factory Module
====================

Factory for creating and configuring models.

Author: BUL System
Date: 2024
"""

import logging
from typing import Dict, Any, Optional
from transformers import PreTrainedModel

from ..device_manager import DeviceManager
from ..model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating model instances.
    
    Provides convenient methods for model creation with different configurations.
    
    Example:
        >>> factory = ModelFactory(device_manager)
        >>> model = factory.create_causal_model("gpt2")
        >>> model = factory.create_seq2seq_model("t5-small")
    """
    
    def __init__(self, device_manager: DeviceManager):
        """
        Initialize ModelFactory.
        
        Args:
            device_manager: DeviceManager instance
        """
        self.device_manager = device_manager
    
    def create_causal_model(
        self,
        model_name: str,
        tokenizer_vocab_size: Optional[int] = None,
        **kwargs
    ) -> PreTrainedModel:
        """
        Create a causal language model.
        
        Args:
            model_name: Name or path of the model
            tokenizer_vocab_size: Vocabulary size for token embeddings
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model
        """
        loader = ModelLoader(
            model_name=model_name,
            model_type="causal",
            device_manager=self.device_manager,
            tokenizer_vocab_size=tokenizer_vocab_size
        )
        return loader.load()
    
    def create_seq2seq_model(
        self,
        model_name: str,
        tokenizer_vocab_size: Optional[int] = None,
        **kwargs
    ) -> PreTrainedModel:
        """
        Create a seq2seq model.
        
        Args:
            model_name: Name or path of the model
            tokenizer_vocab_size: Vocabulary size for token embeddings
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model
        """
        loader = ModelLoader(
            model_name=model_name,
            model_type="seq2seq",
            device_manager=self.device_manager,
            tokenizer_vocab_size=tokenizer_vocab_size
        )
        return loader.load()
    
    def create_from_config(self, config: Dict[str, Any]) -> PreTrainedModel:
        """
        Create model from configuration.
        
        Args:
            config: Configuration dictionary with model_name and model_type
            
        Returns:
            Loaded model
        """
        model_type = config.get("model_type", "causal")
        
        if model_type == "causal":
            return self.create_causal_model(
                model_name=config["model_name"],
                tokenizer_vocab_size=config.get("tokenizer_vocab_size"),
                **{k: v for k, v in config.items() if k not in ["model_name", "model_type", "tokenizer_vocab_size"]}
            )
        else:
            return self.create_seq2seq_model(
                model_name=config["model_name"],
                tokenizer_vocab_size=config.get("tokenizer_vocab_size"),
                **{k: v for k, v in config.items() if k not in ["model_name", "model_type", "tokenizer_vocab_size"]}
            )


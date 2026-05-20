"""
Model Loader Module
===================

Handles loading and configuration of pre-trained language models.
Supports both causal and seq2seq model types.

Author: BUL System
Date: 2024
"""

import logging
import torch
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
)
from .device_manager import DeviceManager

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and configures pre-trained language models.
    
    Supports:
    - Causal language models (GPT-2, GPT-Neo, etc.)
    - Seq2seq models (T5, BART, etc.)
    
    Attributes:
        model: The loaded model
        model_name: Name or path of the model
        model_type: Type of model ("causal" or "seq2seq")
        device_manager: DeviceManager instance
        
    Example:
        >>> loader = ModelLoader("gpt2", "causal", device_manager)
        >>> model = loader.load()
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        device_manager: DeviceManager,
        tokenizer_vocab_size: Optional[int] = None
    ):
        """
        Initialize ModelLoader.
        
        Args:
            model_name: Name or path of the pre-trained model
            model_type: Type of model ("causal" or "seq2seq")
            device_manager: DeviceManager instance
            tokenizer_vocab_size: Vocabulary size of tokenizer (for resizing)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device_manager = device_manager
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.model: Optional[PreTrainedModel] = None
        
        if model_type not in ["causal", "seq2seq"]:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'causal' or 'seq2seq'")
    
    def load(self) -> PreTrainedModel:
        """
        Load the pre-trained model.
        
        Returns:
            Loaded model
            
        Raises:
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model: {self.model_name} (type: {self.model_type})")
        
        try:
            # Determine torch dtype based on device
            torch_dtype = self._get_torch_dtype()
            
            # Load model based on type
            if self.model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device_manager.is_cuda_available() else None,
                )
            else:  # seq2seq
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device_manager.is_cuda_available() else None,
                )
            
            # Resize token embeddings if needed
            if self.tokenizer_vocab_size:
                self._resize_embeddings(model)
            
            # Move to device if not using device_map
            if not self.device_manager.is_cuda_available() or not hasattr(model, 'device'):
                device = self.device_manager.get_device()
                model.to(device)
            
            # Log model info
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Model loaded with {param_count/1e6:.2f}M parameters "
                f"on {self.device_manager.get_device()}"
            )
            
            self.model = model
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model {self.model_name}: {e}")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """
        Get appropriate torch dtype based on device.
        
        Returns:
            torch.dtype to use
        """
        if self.device_manager.is_cuda_available():
            return torch.float16
        elif self.device_manager.is_tpu_available():
            return torch.float32
        else:
            return torch.float32
    
    def _resize_embeddings(self, model: PreTrainedModel) -> None:
        """
        Resize token embeddings if tokenizer vocabulary is larger.
        
        Args:
            model: The model to resize
        """
        if self.tokenizer_vocab_size and self.tokenizer_vocab_size > model.config.vocab_size:
            model.resize_token_embeddings(self.tokenizer_vocab_size)
            logger.info(
                f"Resized token embeddings from {model.config.vocab_size} "
                f"to {self.tokenizer_vocab_size}"
            )
    
    def get_model(self) -> Optional[PreTrainedModel]:
        """Get the loaded model."""
        return self.model
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "vocab_size": self.model.config.vocab_size,
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "parameters_millions": param_count / 1e6,
            "device": str(self.device_manager.get_device()),
        }


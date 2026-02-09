"""
Base Backend Interface - Abstract base class for all model backends.

This module defines the interface that all model backends must implement.
It provides common functionality and validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional, Iterator
from contextlib import contextmanager
import logging

from ..model_loader.types import ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """
    Base interface for model backends.
    
    All backends must implement:
    - load(): Load model and tokenizer
    - generate(): Generate text from prompts
    - unload(): Unload model from memory
    
    Backends can optionally implement:
    - encode(): Encode text to tokens
    - decode(): Decode tokens to text
    - get_model_info(): Get model metadata
    """
    
    def __init__(self, name: str = "base"):
        """
        Initialize base backend.
        
        Args:
            name: Backend name for logging
        """
        self._loaded = False
        self._config: Optional[ModelConfig] = None
        self._name = name
        self._model_info: Dict[str, Any] = {}
    
    @abstractmethod
    def load(self, config: ModelConfig) -> Dict[str, Any]:
        """
        Load model and tokenizer.
        
        Args:
            config: Model configuration
            
        Returns:
            Dictionary with loaded components:
            - model: The loaded model
            - tokenizer: The tokenizer (if applicable)
            - device: Device information
            - other backend-specific components
        
        Raises:
            RuntimeError: If loading fails
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: GenerationConfig
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Single prompt or list of prompts
            config: Generation configuration
            
        Returns:
            Generated text(s) - single string if single prompt,
            list of strings if list of prompts
        
        Raises:
            RuntimeError: If model not loaded or generation fails
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload model from memory.
        
        This should free all model-related resources.
        """
        pass
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Single text or list of texts
            add_special_tokens: Whether to add special tokens
        
        Returns:
            Token IDs - single list if single text,
            list of lists if list of texts
        
        Raises:
            NotImplementedError: If backend doesn't support encoding
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support encoding"
        )
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Single list of token IDs or list of lists
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text(s) - single string if single list,
            list of strings if list of lists
        
        Raises:
            NotImplementedError: If backend doesn't support decoding
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support decoding"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information.
        
        Returns:
            Dictionary with model information:
            - name: Model name
            - size: Model size
            - parameters: Number of parameters
            - device: Device information
            - other backend-specific info
        """
        return self._model_info.copy()
    
    def validate_config(self, config: ModelConfig) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration to validate
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not config.model_path and not config.model_name:
            raise ValueError("Either model_path or model_name must be provided")
        
        if config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if config.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if not (0 < config.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        
        return True
    
    def validate_generation_config(self, config: GenerationConfig) -> bool:
        """
        Validate generation configuration.
        
        Args:
            config: Generation configuration to validate
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        if config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if config.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if not (0 < config.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        
        if config.top_k < 0:
            raise ValueError("top_k must be non-negative")
        
        return True
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def config(self) -> Optional[ModelConfig]:
        """Get current model configuration."""
        return self._config
    
    @property
    def name(self) -> str:
        """Get backend name."""
        return self._name
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        if self._loaded:
            self.unload()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self._loaded else "unloaded"
        return f"{self.__class__.__name__}(name={self._name}, status={status})"

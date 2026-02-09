"""
Model Loader - Main model loading class.

This module provides the high-level ModelLoader class that manages
model loading, generation, and lifecycle across different backends.
"""

import logging
from typing import Optional, Dict, Any, Union, List

from .types import ModelConfig, GenerationConfig, ModelType, QuantizationType, BackendType
from .factory import create_backend
from ..backends.base import BaseBackend

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Advanced model loader with multiple backend support.
    
    Supports:
    - vLLM: Fast inference with PagedAttention
    - Transformers: Standard HuggingFace models
    - llama.cpp: CPU-optimized inference
    - TensorRT-LLM: NVIDIA-optimized inference (future)
    
    Examples:
        >>> # Basic usage
        >>> loader = ModelLoader(
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     backend=BackendType.VLLM
        ... )
        >>> result = loader.load()
        >>> text = loader.generate("Hello, world!")
        
        >>> # With configuration
        >>> config = ModelConfig(
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     quantization=QuantizationType.FP16,
        ...     backend=BackendType.AUTO
        ... )
        >>> loader = ModelLoader.from_config(config)
        >>> result = loader.load()
        
        >>> # Context manager
        >>> with ModelLoader(model_name="model") as loader:
        ...     text = loader.generate("Hello!")
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        model_type: ModelType = ModelType.CAUSAL_LM,
        quantization: QuantizationType = QuantizationType.FP16,
        device: str = "cuda",
        backend: BackendType = BackendType.AUTO,
        use_flash_attention: bool = True,
        trust_remote_code: bool = False,
        **kwargs
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: Model name (HuggingFace ID or local path)
            model_path: Local path to model (optional)
            model_type: Type of model
            quantization: Quantization type
            device: Device to use (cuda/cpu)
            backend: Backend to use (auto/vllm/transformers/llama_cpp)
            use_flash_attention: Use Flash Attention
            trust_remote_code: Trust remote code
            **kwargs: Additional arguments
        """
        self.config = ModelConfig(
            model_name=model_name,
            model_path=model_path,
            model_type=model_type,
            quantization=quantization,
            device=device,
            backend=backend,
            use_flash_attention=use_flash_attention,
            trust_remote_code=trust_remote_code,
            extra_kwargs=kwargs
        )
        
        self.backend: Optional[BaseBackend] = None
        self._loaded_components: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> "ModelLoader":
        """
        Create ModelLoader from ModelConfig.
        
        Args:
            config: Model configuration
        
        Returns:
            ModelLoader instance
        """
        loader = cls.__new__(cls)
        loader.config = config
        loader.backend = None
        loader._loaded_components = None
        return loader
    
    def load(self) -> Dict[str, Any]:
        """
        Load model and tokenizer.
        
        Returns:
            Dictionary with 'model', 'tokenizer', 'processor' (if applicable), and 'backend'
        
        Raises:
            RuntimeError: If backend is not available or loading fails
        """
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Create backend
        self.backend = create_backend(self.config.backend)
        
        # Load model
        self._loaded_components = self.backend.load(self.config)
        
        logger.info(
            f"Model loaded successfully with backend: {self._loaded_components['backend']}"
        )
        
        return self._loaded_components
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text(s)
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.backend or not self.backend.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        gen_config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            extra_kwargs=kwargs
        )
        
        return self.backend.generate(prompt, gen_config)
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.backend:
            self.backend.unload()
            self.backend = None
        self._loaded_components = None
        logger.info("Model unloaded from memory")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.backend is not None and self.backend.is_loaded
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"ModelLoader(model={self.config.model_name}, "
            f"backend={self.config.backend}, status={status})"
        )


__all__ = [
    "ModelLoader",
]













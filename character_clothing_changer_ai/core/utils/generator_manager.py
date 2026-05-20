"""
Generator Manager Utility
=========================

Utility for managing ComfyUI tensor generator initialization and lifecycle.
"""

import logging
from typing import Optional
from pathlib import Path

from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.comfyui_tensor_generator import ComfyUITensorGenerator
from ..utils.model_initializer import ModelInitializer

logger = logging.getLogger(__name__)


class GeneratorManager:
    """Manages ComfyUI tensor generator initialization and lifecycle."""
    
    @staticmethod
    def ensure_generator_initialized(
        generator: Optional[ComfyUITensorGenerator],
        model: Optional[Flux2ClothingChangerModel],
        output_dir: str,
        initialize_model_fn=None
    ) -> ComfyUITensorGenerator:
        """
        Ensure generator is initialized, initialize if needed.
        
        Args:
            generator: Current generator instance (may be None)
            model: Flux2 model instance (may be None)
            output_dir: Output directory for tensors
            initialize_model_fn: Optional function to initialize model if None
            
        Returns:
            Initialized generator
        """
        if generator is not None:
            return generator
        
        # Ensure model is initialized first
        if model is None:
            if initialize_model_fn:
                initialize_model_fn()
            else:
                raise RuntimeError("Model must be initialized before generator")
        
        # Initialize generator using ModelInitializer
        return ModelInitializer.initialize_generator(model, output_dir)
    
    @staticmethod
    def can_generate_tensors(
        model: Optional[Flux2ClothingChangerModel],
        use_deepseek_fallback: bool = False
    ) -> bool:
        """
        Check if tensor generation is possible.
        
        Args:
            model: Flux2 model instance
            use_deepseek_fallback: Whether using DeepSeek fallback
            
        Returns:
            True if tensor generation is possible
        """
        return model is not None and not use_deepseek_fallback
    
    @staticmethod
    def cleanup_generator(generator: Optional[ComfyUITensorGenerator]) -> None:
        """
        Clean up generator resources.
        
        Args:
            generator: Generator instance to cleanup
        """
        if generator is not None:
            try:
                del generator
                logger.debug("Generator deleted")
            except Exception as e:
                logger.warning(f"Error deleting generator: {e}")


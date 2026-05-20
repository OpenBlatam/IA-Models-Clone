"""
Resource Cleaner
================

Handles cleanup of service resources, including models,
generators, and other components.
"""

import logging
from typing import Optional

from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.deepseek_clothing_model import DeepSeekClothingModel
from ...models.comfyui_tensor_generator import ComfyUITensorGenerator
from ...core.utils.model_initializer import ModelInitializer
from ...core.utils.generator_manager import GeneratorManager

logger = logging.getLogger(__name__)


class ResourceCleaner:
    """Handles resource cleanup operations."""
    
    @staticmethod
    def cleanup_model(model: Optional[Flux2ClothingChangerModel]) -> None:
        """
        Cleanup model resources.
        
        Args:
            model: Model instance to cleanup
        """
        if model is not None:
            ModelInitializer.cleanup_resources(model)
            logger.debug("Model resources cleaned up")
    
    @staticmethod
    def cleanup_deepseek_model(model: Optional[DeepSeekClothingModel]) -> None:
        """
        Cleanup DeepSeek model resources.
        
        Args:
            model: DeepSeek model instance to cleanup
        """
        if model is not None:
            del model
            logger.debug("DeepSeek model resources cleaned up")
    
    @staticmethod
    def cleanup_generator(generator: Optional[ComfyUITensorGenerator]) -> None:
        """
        Cleanup generator resources.
        
        Args:
            generator: Generator instance to cleanup
        """
        if generator is not None:
            GeneratorManager.cleanup_generator(generator)
            logger.debug("Generator resources cleaned up")
    
    @staticmethod
    def cleanup_all(
        model: Optional[Flux2ClothingChangerModel],
        deepseek_model: Optional[DeepSeekClothingModel],
        generator: Optional[ComfyUITensorGenerator]
    ) -> None:
        """
        Cleanup all resources.
        
        Args:
            model: Flux2 model instance
            deepseek_model: DeepSeek model instance
            generator: Generator instance
        """
        ResourceCleaner.cleanup_model(model)
        ResourceCleaner.cleanup_generator(generator)
        ResourceCleaner.cleanup_deepseek_model(deepseek_model)
        logger.info("All service resources cleaned up")


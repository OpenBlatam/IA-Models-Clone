"""
Service Initializer
===================

Handles service initialization, including model setup,
generator initialization, and component configuration.
"""

import logging
from typing import Optional, Tuple

from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.deepseek_clothing_model import DeepSeekClothingModel
from ...models.comfyui_tensor_generator import ComfyUITensorGenerator
from ...config.clothing_changer_config import ClothingChangerConfig
from ...core.factories.model_factory import ModelFactory
from ...core.utils.model_selector import ModelSelector
from ...core.utils.generator_manager import GeneratorManager
from ...core.utils.model_initializer import ModelInitializer

logger = logging.getLogger(__name__)


class ServiceInitializer:
    """Handles service initialization operations."""
    
    def __init__(self, config: ClothingChangerConfig):
        """
        Initialize Service Initializer.
        
        Args:
            config: Configuration instance
        """
        self.config = config
    
    def initialize_models(
        self,
        existing_model: Optional[Flux2ClothingChangerModel] = None,
        existing_deepseek: Optional[DeepSeekClothingModel] = None
    ) -> Tuple[
        Optional[Flux2ClothingChangerModel],
        Optional[DeepSeekClothingModel],
        bool
    ]:
        """
        Initialize models using factory.
        
        Args:
            existing_model: Existing Flux2 model (if any)
            existing_deepseek: Existing DeepSeek model (if any)
            
        Returns:
            Tuple of (model, deepseek_model, use_deepseek_fallback)
        """
        if existing_model is not None or existing_deepseek is not None:
            logger.warning("Model already initialized")
            return existing_model, existing_deepseek, False
        
        # Use factory to create models
        model, deepseek_model, use_deepseek_fallback = ModelFactory.create_models(
            self.config
        )
        
        return model, deepseek_model, use_deepseek_fallback
    
    def initialize_generator(
        self,
        model: Optional[Flux2ClothingChangerModel]
    ) -> Optional[ComfyUITensorGenerator]:
        """
        Initialize generator if model is available.
        
        Args:
            model: Flux2 model instance
            
        Returns:
            Generator instance or None
        """
        if model:
            generator = ModelInitializer.initialize_generator(
                model,
                self.config.output_dir
            )
            logger.info("Flux2 model and generator initialized successfully")
            return generator
        return None
    
    def create_model_selector(
        self,
        model: Optional[Flux2ClothingChangerModel],
        deepseek_model: Optional[DeepSeekClothingModel],
        use_deepseek_fallback: bool
    ) -> ModelSelector:
        """
        Create model selector instance.
        
        Args:
            model: Flux2 model instance
            deepseek_model: DeepSeek model instance
            use_deepseek_fallback: Whether using DeepSeek fallback
            
        Returns:
            ModelSelector instance
        """
        return ModelSelector(
            flux2_model=model,
            deepseek_model=deepseek_model,
            use_deepseek_fallback=use_deepseek_fallback,
            config=self.config
        )
    
    def create_generator_manager(
        self,
        generator: Optional[ComfyUITensorGenerator],
        model: Optional[Flux2ClothingChangerModel],
        initialize_model_fn: callable
    ) -> GeneratorManager:
        """
        Create generator manager instance.
        
        Args:
            generator: Generator instance
            model: Model instance
            initialize_model_fn: Function to initialize model
            
        Returns:
            GeneratorManager instance
        """
        return GeneratorManager(
            generator=generator,
            config=self.config,
            model=model,
            initialize_model_fn=initialize_model_fn
        )


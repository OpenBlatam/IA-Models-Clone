"""
Model Manager
============

Handles model initialization and management.
"""

import logging
from typing import Optional, Dict, Any

from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.comfyui_tensor_generator import ComfyUITensorGenerator
from ...config.clothing_changer_config import ClothingChangerConfig
from ..utils.model_initializer import ModelInitializer

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model initialization and lifecycle."""
    
    def __init__(self, config: ClothingChangerConfig):
        """
        Initialize Model Manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.model: Optional[Flux2ClothingChangerModel] = None
        self.generator: Optional[ComfyUITensorGenerator] = None
    
    def initialize_model(self) -> None:
        """Initialize the Flux2 model."""
        if self.model is not None:
            logger.warning("Model already initialized")
            return
        
        # Use ModelInitializer for consistent initialization
        self.model = ModelInitializer.initialize_flux2_model(self.config)
        self.generator = ModelInitializer.initialize_generator(
            self.model,
            self.config.output_dir
        )
    
    def ensure_model_initialized(self) -> None:
        """Ensure model is initialized, initialize if not."""
        ModelInitializer.ensure_model_initialized(
            self.model,
            self.initialize_model
        )
    
    def ensure_generator_initialized(self) -> None:
        """Ensure generator is initialized, initialize if not."""
        if self.generator is None:
            self.ensure_model_initialized()
            self.generator = ModelInitializer.initialize_generator(
                self.model,
                self.config.output_dir
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return ModelInitializer.get_model_info(self.model)
    
    def close(self) -> None:
        """Clean up resources."""
        ModelInitializer.cleanup_resources(self.model)
        self.model = None
        self.generator = None
        logger.info("Model resources cleaned up")



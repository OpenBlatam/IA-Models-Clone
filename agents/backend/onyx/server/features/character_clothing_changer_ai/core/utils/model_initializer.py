"""
Model Initializer Utility
=========================

Utility for initializing models with consistent error handling.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import torch

from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.deepseek_clothing_model import DeepSeekClothingModel
from ...models.comfyui_tensor_generator import ComfyUITensorGenerator
from ...config.clothing_changer_config import ClothingChangerConfig
from ..exceptions import ModelLoadError, ModelNotInitializedError

logger = logging.getLogger(__name__)


class ModelInitializer:
    """Handles model initialization with consistent error handling."""
    
    @staticmethod
    def get_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
        """
        Convert dtype string to torch dtype.
        
        Args:
            dtype_str: Dtype string ('float16', 'float32', etc.)
            
        Returns:
            torch dtype or None
        """
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "float32":
            return torch.float32
        return None
    
    @staticmethod
    def initialize_flux2_model(
        config: ClothingChangerConfig
    ) -> Flux2ClothingChangerModel:
        """
        Initialize Flux2 model.
        
        Args:
            config: Configuration instance
            
        Returns:
            Initialized Flux2 model
            
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        logger.info("Initializing Flux2 Clothing Changer Model...")
        
        try:
            dtype = ModelInitializer.get_dtype(config.dtype)
            
            model = Flux2ClothingChangerModel(
                model_id=config.model_id,
                device=config.device,
                dtype=dtype,
                enable_optimizations=config.enable_optimizations,
                use_inpainting=config.use_inpainting,
                use_controlnet=config.use_controlnet,
            )
            
            logger.info("Flux2 model initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing Flux2 model: {e}")
            raise ModelLoadError(
                model_id=config.model_id,
                reason=str(e)
            )
    
    @staticmethod
    def initialize_generator(
        model: Flux2ClothingChangerModel,
        output_dir: str
    ) -> ComfyUITensorGenerator:
        """
        Initialize ComfyUI tensor generator.
        
        Args:
            model: Flux2 model instance
            output_dir: Output directory for tensors
            
        Returns:
            Initialized generator
        """
        logger.info("Initializing ComfyUI tensor generator...")
        
        try:
            generator = ComfyUITensorGenerator(
                model=model,
                output_dir=output_dir,
            )
            
            logger.info("Generator initialized successfully")
            return generator
            
        except Exception as e:
            logger.error(f"Error initializing generator: {e}")
            raise RuntimeError(f"Cannot initialize generator: {e}")
    
    @staticmethod
    def ensure_model_initialized(
        model: Optional[Flux2ClothingChangerModel],
        initialize_fn
    ) -> None:
        """
        Ensure model is initialized, raise error if not.
        
        Args:
            model: Model instance to check
            initialize_fn: Function to call if model is None
            
        Raises:
            ModelNotInitializedError: If model is not initialized
        """
        if model is None:
            try:
                initialize_fn()
            except Exception as e:
                raise ModelNotInitializedError() from e
    
    @staticmethod
    def cleanup_resources(model: Optional[Any] = None) -> None:
        """
        Clean up model resources.
        
        Args:
            model: Model instance to cleanup
        """
        if model is not None:
            try:
                del model
                logger.debug("Model deleted")
            except Exception as e:
                logger.warning(f"Error deleting model: {e}")
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")
    
    @staticmethod
    def get_model_info(model: Optional[Flux2ClothingChangerModel]) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model: Model instance
            
        Returns:
            Model info dictionary
        """
        if model is None:
            return {"status": "not_initialized"}
        
        try:
            return model.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


"""
Model Factory
=============

Factory pattern for creating and initializing AI models with fallback support.
"""

import logging
import os
from typing import Optional, Tuple, Any
from ..clothing_changer_service import ClothingChangerService
from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.deepseek_clothing_model import DeepSeekClothingModel
from ...config.clothing_changer_config import ClothingChangerConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and initializing AI models with automatic fallback."""
    
    @staticmethod
    def create_models(
        config: ClothingChangerConfig
    ) -> Tuple[Optional[Flux2ClothingChangerModel], Optional[DeepSeekClothingModel], bool]:
        """
        Create and initialize models with automatic fallback.
        
        Args:
            config: Configuration instance
            
        Returns:
            Tuple of (flux2_model, deepseek_model, use_deepseek_fallback)
        """
        flux2_model = None
        deepseek_model = None
        use_fallback = False
        
        # Try Flux2 first
        try:
            logger.info("Initializing Flux2 Clothing Changer Model...")
            flux2_model = Flux2ClothingChangerModel(
                model_id=config.model_id,
                device=config.device,
                dtype=ModelFactory._get_dtype(config.dtype),
                enable_optimizations=config.enable_optimizations,
                use_inpainting=config.use_inpainting,
                use_controlnet=config.use_controlnet,
            )
            logger.info("Flux2 model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Flux2 model: {e}")
            logger.info("Falling back to DeepSeek model...")
            
            # Try DeepSeek fallback
            try:
                api_key = os.getenv("DEEPSEEK_API_KEY", "sk-753365753f074509bb52496e038691f6")
                deepseek_model = DeepSeekClothingModel(api_key=api_key)
                use_fallback = True
                logger.info("DeepSeek model initialized as fallback")
                
            except Exception as deepseek_error:
                logger.error(f"Error initializing DeepSeek fallback: {deepseek_error}")
                raise RuntimeError(
                    f"Failed to initialize both Flux2 and DeepSeek models. "
                    f"Flux2 error: {e}. DeepSeek error: {deepseek_error}"
                )
        
        return flux2_model, deepseek_model, use_fallback
    
    @staticmethod
    def _get_dtype(dtype_str: str):
        """Convert dtype string to torch dtype."""
        import torch
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "float32":
            return torch.float32
        return None


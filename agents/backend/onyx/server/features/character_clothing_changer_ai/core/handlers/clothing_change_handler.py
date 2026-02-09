"""
Clothing Change Handler
=======================

Handles clothing change operations.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import numpy as np

from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...config.clothing_changer_config import ClothingChangerConfig
from ..utils.image_loader import ImageLoader
from ..utils.image_saver import ImageSaver

logger = logging.getLogger(__name__)


class ClothingChangeHandler:
    """Handles clothing change operations."""
    
    def __init__(
        self,
        model: Flux2ClothingChangerModel,
        config: ClothingChangerConfig
    ):
        """
        Initialize Clothing Change Handler.
        
        Args:
            model: Flux2 model instance
            config: Configuration instance
        """
        self.model = model
        self.config = config
    
    def change_clothing(
        self,
        image: Union[str, Path, Image.Image],
        clothing_description: str,
        character_name: Optional[str] = None,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ) -> Image.Image:
        """
        Change clothing in character image.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            character_name: Optional character name
            mask: Optional mask for clothing area
            prompt: Optional full prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            
        Returns:
            Changed image
        """
        logger.info(f"Changing clothing: {clothing_description}")
        
        changed_image = self.model.change_clothing(
            image=image,
            clothing_description=clothing_description,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt or self.config.default_negative_prompt,
            num_inference_steps=num_inference_steps or self.config.default_num_inference_steps,
            guidance_scale=guidance_scale or self.config.default_guidance_scale,
            strength=strength or self.config.default_strength,
        )
        
        return changed_image
    
    def load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load image from various formats.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            PIL Image
        """
        return ImageLoader.load(image)
    
    def save_temp_image(self, image: Image.Image, character_name: Optional[str] = None) -> Path:
        """
        Save image to temporary location.
        
        Args:
            image: Image to save
            character_name: Optional character name for filename
            
        Returns:
            Path to saved image
        """
        return ImageSaver.save_temp(
            image=image,
            prefix="clothing_change",
            suffix=character_name
        )



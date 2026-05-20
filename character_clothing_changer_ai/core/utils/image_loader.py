"""
Image Loader Utility
====================

Utility for loading images from various formats.
"""

import logging
from pathlib import Path
from typing import Union
from PIL import Image

from .image_validator_unified import UnifiedImageValidator

logger = logging.getLogger(__name__)


class ImageLoader:
    """Handles loading images from various formats."""
    
    @staticmethod
    def load(image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load image from various formats.
        
        Args:
            image: Image path (str/Path) or PIL Image
            
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValueError: If image cannot be loaded
        """
        if isinstance(image, (str, Path)):
            try:
                # Validate path first
                is_valid, error_msg = UnifiedImageValidator.validate_image_path(image)
                if not is_valid:
                    raise ValueError(error_msg or f"Cannot load image from {image}")
                
                img = Image.open(image)
                # Validate and convert
                img = UnifiedImageValidator.validate_and_convert(img)
                logger.debug(f"Loaded image from path: {image}")
                return img
            except Exception as e:
                logger.error(f"Error loading image from {image}: {e}")
                raise ValueError(f"Cannot load image from {image}: {e}")
        elif isinstance(image, Image.Image):
            # Validate and convert
            return UnifiedImageValidator.validate_and_convert(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @staticmethod
    def validate(image: Image.Image) -> bool:
        """
        Validate image is valid for processing.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            True if valid
        """
        is_valid, _ = UnifiedImageValidator.validate_pil_image(image)
        return is_valid


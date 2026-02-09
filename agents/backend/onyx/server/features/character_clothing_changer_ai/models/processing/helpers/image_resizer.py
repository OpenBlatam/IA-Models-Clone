"""
Image Resizer Helper
====================

Handles image resizing with validation.
"""

import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImageResizer:
    """Handles image resizing operations."""
    
    @staticmethod
    def resize_to_max(
        image: Image.Image,
        max_size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """
        Resize image if it exceeds maximum size.
        
        Args:
            image: PIL Image to resize
            max_size: Maximum dimension
            resample: Resampling method
            
        Returns:
            Resized image (or original if no resize needed)
        """
        if max(image.size) <= max_size:
            return image
        
        ratio = max_size / max(image.size)
        new_size = (
            int(image.size[0] * ratio),
            int(image.size[1] * ratio)
        )
        resized = image.resize(new_size, resample)
        logger.debug(f"Resized image to {new_size} (max_size={max_size})")
        return resized
    
    @staticmethod
    def resize_to_min(
        image: Image.Image,
        min_size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """
        Upscale image if it's below minimum size.
        
        Args:
            image: PIL Image to resize
            min_size: Minimum dimension
            resample: Resampling method
            
        Returns:
            Resized image (or original if no resize needed)
        """
        if min(image.size) >= min_size:
            return image
        
        ratio = min_size / min(image.size)
        new_size = (
            int(image.size[0] * ratio),
            int(image.size[1] * ratio)
        )
        resized = image.resize(new_size, resample)
        logger.debug(f"Upscaled small image to {new_size} (min_size={min_size})")
        return resized
    
    @staticmethod
    def validate_and_resize(
        image: Image.Image,
        max_size: int,
        min_size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """
        Validate and resize image to fit constraints.
        
        Args:
            image: PIL Image to process
            max_size: Maximum dimension
            min_size: Minimum dimension
            resample: Resampling method
            
        Returns:
            Validated and resized image
        """
        # Validate dimensions
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Image has zero dimensions")
        
        # Resize if too large
        image = ImageResizer.resize_to_max(image, max_size, resample)
        
        # Upscale if too small
        image = ImageResizer.resize_to_min(image, min_size, resample)
        
        return image



"""
Image Converter Helper
======================

Converts various image formats to PIL Image.
"""

import logging
from typing import Union
from pathlib import Path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageConverter:
    """Converts various image formats to PIL Image."""
    
    @staticmethod
    def to_pil_image(image: Union[Image.Image, str, Path, np.ndarray]) -> Image.Image:
        """
        Convert various image formats to PIL Image with validation.
        
        Args:
            image: Input image in various formats
            
        Returns:
            PIL Image in RGB mode
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is unsupported or invalid
        """
        try:
            if isinstance(image, (str, Path)):
                return ImageConverter._from_path(image)
            elif isinstance(image, np.ndarray):
                return ImageConverter._from_numpy(image)
            elif isinstance(image, Image.Image):
                return ImageConverter._from_pil(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        except Exception as e:
            logger.error(f"Error converting image: {e}")
            raise ValueError(f"Failed to process image: {e}") from e
    
    @staticmethod
    def _from_path(path: Union[str, Path]) -> Image.Image:
        """Convert from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil_image = Image.open(path).convert("RGB")
        ImageConverter._validate_dimensions(pil_image)
        return pil_image
    
    @staticmethod
    def _from_numpy(array: np.ndarray) -> Image.Image:
        """Convert from numpy array."""
        if array.size == 0:
            raise ValueError("Empty numpy array")
        
        # Normalize dtype
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        
        pil_image = Image.fromarray(array).convert("RGB")
        ImageConverter._validate_dimensions(pil_image)
        return pil_image
    
    @staticmethod
    def _from_pil(image: Image.Image) -> Image.Image:
        """Convert from PIL Image."""
        pil_image = image.convert("RGB")
        ImageConverter._validate_dimensions(pil_image)
        return pil_image
    
    @staticmethod
    def _validate_dimensions(image: Image.Image) -> None:
        """Validate image dimensions."""
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Invalid image dimensions")



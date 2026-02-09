"""
Utility Mixin

Contains utility methods and helpers.
"""

import logging
from typing import Union, Tuple, Optional, Dict, Any, List
from pathlib import Path
from PIL import Image

from ..helpers import (
    DimensionCalculator,
    ImageProcessingUtils,
)

logger = logging.getLogger(__name__)


class UtilityMixin:
    """
    Mixin providing utility functionality.
    
    This mixin contains:
    - Image utilities
    - Dimension utilities
    - Format conversion
    - File operations
    - Helper methods
    """
    
    def get_optimal_resolution(
        self,
        original_size: Tuple[int, int],
        scale_factor: float,
        max_dimension: Optional[int] = None,
        maintain_aspect: bool = True
    ) -> Tuple[int, int]:
        """
        Calculate optimal resolution for upscaling.
        
        Args:
            original_size: Original image size (width, height)
            scale_factor: Scale factor
            max_dimension: Maximum dimension (width or height)
            maintain_aspect: Maintain aspect ratio
            
        Returns:
            Tuple of (width, height)
        """
        return DimensionCalculator.calculate_optimal_resolution(
            original_size,
            scale_factor,
            max_dimension=max_dimension,
            maintain_aspect=maintain_aspect
        )
    
    def resize_to_fit(
        self,
        image: Image.Image,
        max_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image to fit within maximum size.
        
        Args:
            image: Input image
            max_size: Maximum size (width, height)
            maintain_aspect: Maintain aspect ratio
            
        Returns:
            Resized image
        """
        return ImageProcessingUtils.resize_to_fit(image, max_size, maintain_aspect)
    
    def convert_format(
        self,
        image: Image.Image,
        target_format: str = "RGB"
    ) -> Image.Image:
        """
        Convert image format.
        
        Args:
            image: Input image
            target_format: Target format (RGB, RGBA, L, etc.)
            
        Returns:
            Converted image
        """
        return ImageProcessingUtils.convert_format(image, target_format)
    
    def get_image_info(
        self,
        image: Union[Image.Image, str, Path]
    ) -> Dict[str, Any]:
        """
        Get comprehensive image information.
        
        Args:
            image: Input image or path
            
        Returns:
            Dictionary with image information
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            file_path = str(image)
        else:
            pil_image = image.convert("RGB")
            file_path = None
        
        return {
            "size": pil_image.size,
            "mode": pil_image.mode,
            "format": pil_image.format if hasattr(pil_image, 'format') else None,
            "file_path": file_path,
            "width": pil_image.size[0],
            "height": pil_image.size[1],
            "aspect_ratio": pil_image.size[0] / pil_image.size[1] if pil_image.size[1] > 0 else 0,
            "total_pixels": pil_image.size[0] * pil_image.size[1],
        }
    
    def validate_image_file(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Validate image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "is_valid": False,
                "errors": [f"File does not exist: {file_path}"],
                "warnings": []
            }
        
        try:
            with Image.open(file_path) as img:
                img.verify()
            
            return {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_size": file_path.stat().st_size,
                "file_path": str(file_path)
            }
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Invalid image file: {str(e)}"],
                "warnings": [],
                "file_path": str(file_path)
            }
    
    def batch_get_image_info(
        self,
        images: List[Union[Image.Image, str, Path]]
    ) -> List[Dict[str, Any]]:
        """
        Get information for multiple images.
        
        Args:
            images: List of images or paths
            
        Returns:
            List of image information dictionaries
        """
        return [self.get_image_info(img) for img in images]
    
    def create_thumbnail(
        self,
        image: Union[Image.Image, str, Path],
        size: Tuple[int, int] = (128, 128),
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Create thumbnail from image.
        
        Args:
            image: Input image or path
            size: Thumbnail size (width, height)
            maintain_aspect: Maintain aspect ratio
            
        Returns:
            Thumbnail image
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if maintain_aspect:
            pil_image.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
        
        return pil_image



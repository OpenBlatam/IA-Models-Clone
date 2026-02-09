"""
Image Validator Helper
======================

Validates image properties and dimensions.
"""

import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validates image properties."""
    
    @staticmethod
    def validate_dimensions(image: Image.Image) -> bool:
        """
        Validate image dimensions.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Image has zero dimensions")
        return True
    
    @staticmethod
    def validate_size(
        image: Image.Image,
        min_size: int = 64,
        max_size: int = 2048
    ) -> tuple[bool, str]:
        """
        Validate image size constraints.
        
        Args:
            image: PIL Image to validate
            min_size: Minimum dimension
            max_size: Maximum dimension
            
        Returns:
            Tuple of (is_valid, message)
        """
        width, height = image.size
        max_dim = max(width, height)
        min_dim = min(width, height)
        
        if max_dim > max_size:
            return False, f"Image too large: {max_dim} > {max_size}"
        
        if min_dim < min_size:
            return False, f"Image too small: {min_dim} < {min_size}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_format(image: Image.Image) -> bool:
        """
        Validate image format.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            True if valid format
        """
        if image.mode not in ("RGB", "RGBA", "L", "P"):
            logger.warning(f"Image mode {image.mode} may need conversion to RGB")
        return True



"""
Image Validator

Validates image data and metadata.
"""

import numpy as np
from typing import Tuple, Optional
from ..exceptions import InvalidImageException


class ImageValidator:
    """
    Validator for image data.
    
    Validates image dimensions, format, and content.
    """
    
    MIN_WIDTH = 32
    MIN_HEIGHT = 32
    MAX_WIDTH = 10000
    MAX_HEIGHT = 10000
    SUPPORTED_CHANNELS = [1, 3, 4]
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Validate image array.
        
        Args:
            image: Image array to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if numpy array
        if not isinstance(image, np.ndarray):
            return False, "Image must be a numpy array"
        
        # Check dimensions
        if len(image.shape) < 2:
            return False, "Image must have at least 2 dimensions"
        
        height, width = image.shape[:2]
        
        # Check width
        if width < self.MIN_WIDTH:
            return False, f"Image width must be at least {self.MIN_WIDTH} pixels"
        if width > self.MAX_WIDTH:
            return False, f"Image width must be at most {self.MAX_WIDTH} pixels"
        
        # Check height
        if height < self.MIN_HEIGHT:
            return False, f"Image height must be at least {self.MIN_HEIGHT} pixels"
        if height > self.MAX_HEIGHT:
            return False, f"Image height must be at most {self.MAX_HEIGHT} pixels"
        
        # Check channels
        if len(image.shape) == 3:
            channels = image.shape[2]
            if channels not in self.SUPPORTED_CHANNELS:
                return False, f"Unsupported number of channels: {channels}. Must be one of {self.SUPPORTED_CHANNELS}"
        
        # Check data type
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            return False, f"Unsupported data type: {image.dtype}. Must be uint8, float32, or float64"
        
        # Check if image is empty
        if image.size == 0:
            return False, "Image is empty"
        
        # Check if all values are NaN or Inf
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return False, "Image contains NaN or Inf values"
        
        return True, None
    
    def validate_dimensions(
        self,
        width: int,
        height: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate image dimensions.
        
        Args:
            width: Image width
            height: Image height
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if width < self.MIN_WIDTH:
            return False, f"Width must be at least {self.MIN_WIDTH}"
        if width > self.MAX_WIDTH:
            return False, f"Width must be at most {self.MAX_WIDTH}"
        if height < self.MIN_HEIGHT:
            return False, f"Height must be at least {self.MIN_HEIGHT}"
        if height > self.MAX_HEIGHT:
            return False, f"Height must be at most {self.MAX_HEIGHT}"
        
        return True, None
    
    def validate_format(self, image_format: str) -> Tuple[bool, Optional[str]]:
        """
        Validate image format string.
        
        Args:
            image_format: Format string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_formats = ["numpy", "bytes", "file_path", "base64"]
        if image_format not in valid_formats:
            return False, f"Invalid format: {image_format}. Must be one of {valid_formats}"
        return True, None




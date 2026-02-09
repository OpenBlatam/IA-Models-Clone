"""
Unified Image Validator
=======================

Unified image validation utility combining ImageValidator and ImageLoader validation.
"""

import logging
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image
import io

from ..constants import (
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE,
    SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_FILE_SIZE,
    ERROR_IMAGE_NOT_PROVIDED,
    ERROR_INVALID_IMAGE_FORMAT,
    ERROR_IMAGE_TOO_LARGE,
)
from ..exceptions import ImageValidationError

logger = logging.getLogger(__name__)


class UnifiedImageValidator:
    """Unified image validation combining multiple validation approaches."""
    
    @staticmethod
    def validate_image_file(
        image_bytes: bytes,
        filename: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate image file bytes.
        
        Args:
            image_bytes: Image file bytes
            filename: Optional filename for format detection
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not image_bytes:
            return False, ERROR_IMAGE_NOT_PROVIDED
        
        # Check file size
        if len(image_bytes) > MAX_IMAGE_FILE_SIZE:
            return False, ERROR_IMAGE_TOO_LARGE
        
        # Try to open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return UnifiedImageValidator._validate_pil_image(image)
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def validate_image_path(image_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Validate image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(image_path)
        
        if not path.exists():
            return False, f"Image file not found: {image_path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {image_path}"
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > MAX_IMAGE_FILE_SIZE:
            return False, ERROR_IMAGE_TOO_LARGE
        
        # Try to open and validate image
        try:
            image = Image.open(path)
            return UnifiedImageValidator._validate_pil_image(image)
        except Exception as e:
            logger.error(f"Error validating image at {image_path}: {e}")
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def validate_pil_image(image: Image.Image) -> Tuple[bool, Optional[str]]:
        """
        Validate PIL Image object.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, ERROR_IMAGE_NOT_PROVIDED
        
        return UnifiedImageValidator._validate_pil_image(image)
    
    @staticmethod
    def _validate_pil_image(image: Image.Image) -> Tuple[bool, Optional[str]]:
        """
        Internal method to validate PIL Image.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if image has size attribute
        if not hasattr(image, 'size'):
            return False, "Image does not have size attribute"
        
        # Check dimensions
        width, height = image.size
        if width <= 0 or height <= 0:
            return False, f"Invalid image dimensions: {width}x{height}"
        
        max_dimension = max(width, height)
        min_dimension = min(width, height)
        
        if max_dimension > MAX_IMAGE_SIZE:
            return False, f"Image dimensions ({width}x{height}) exceed maximum size of {MAX_IMAGE_SIZE}px"
        
        if min_dimension < MIN_IMAGE_SIZE:
            return False, f"Image dimensions ({width}x{height}) are below minimum size of {MIN_IMAGE_SIZE}px"
        
        # Check format
        if hasattr(image, 'format') and image.format:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                return False, ERROR_INVALID_IMAGE_FORMAT
        
        return True, None
    
    @staticmethod
    def validate_and_convert(image: Image.Image) -> Image.Image:
        """
        Validate and convert image to RGB if needed.
        
        Args:
            image: PIL Image to validate and convert
            
        Returns:
            Validated and converted RGB image
            
        Raises:
            ImageValidationError: If image is invalid
        """
        is_valid, error_msg = UnifiedImageValidator.validate_pil_image(image)
        if not is_valid:
            raise ImageValidationError(error_msg or "Invalid image")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert("RGB")
        
        return image


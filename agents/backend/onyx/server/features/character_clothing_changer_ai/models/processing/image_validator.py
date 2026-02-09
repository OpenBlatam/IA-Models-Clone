"""
Image Validator for Flux2 Clothing Changer
===========================================

Validates image quality and suitability for clothing change operations.
"""

import torch
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validates images for clothing change operations."""
    
    def __init__(
        self,
        min_resolution: Tuple[int, int] = (256, 256),
        max_resolution: Tuple[int, int] = (2048, 2048),
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
        min_brightness: float = 0.1,
        max_brightness: float = 0.95,
        min_contrast: float = 0.2,
        min_sharpness: float = 0.1,
    ):
        """
        Initialize image validator.
        
        Args:
            min_resolution: Minimum image resolution (width, height)
            max_resolution: Maximum image resolution (width, height)
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
            min_brightness: Minimum brightness (0.0 to 1.0)
            max_brightness: Maximum brightness (0.0 to 1.0)
            min_contrast: Minimum contrast (0.0 to 1.0)
            min_sharpness: Minimum sharpness (0.0 to 1.0)
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
    
    def validate(
        self,
        image: Image.Image,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate image for clothing change.
        
        Args:
            image: Image to validate
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }
        
        # Check resolution
        width, height = image.size
        min_w, min_h = self.min_resolution
        max_w, max_h = self.max_resolution
        
        if width < min_w or height < min_h:
            validation_info["valid"] = False
            validation_info["errors"].append(
                f"Resolution too small: {width}x{height} (minimum: {min_w}x{min_h})"
            )
        elif width > max_w or height > max_h:
            validation_info["warnings"].append(
                f"Resolution very large: {width}x{height} (maximum: {max_w}x{max_h})"
            )
        
        validation_info["metrics"]["resolution"] = (width, height)
        
        # Check aspect ratio
        aspect_ratio = width / height
        validation_info["metrics"]["aspect_ratio"] = aspect_ratio
        
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            validation_info["valid"] = False
            validation_info["errors"].append(
                f"Aspect ratio out of range: {aspect_ratio:.2f} "
                f"(range: {self.min_aspect_ratio:.2f} to {self.max_aspect_ratio:.2f})"
            )
        
        # Check image mode
        if image.mode not in ["RGB", "RGBA", "L"]:
            validation_info["warnings"].append(
                f"Unusual image mode: {image.mode} (recommended: RGB)"
            )
        
        # Convert to RGB for analysis
        if image.mode != "RGB":
            image_rgb = image.convert("RGB")
        else:
            image_rgb = image
        
        # Calculate image metrics
        img_array = np.array(image_rgb)
        
        # Brightness (mean of all channels)
        brightness = np.mean(img_array) / 255.0
        validation_info["metrics"]["brightness"] = brightness
        
        if brightness < self.min_brightness:
            validation_info["valid"] = False
            validation_info["errors"].append(
                f"Image too dark: brightness {brightness:.2f} "
                f"(minimum: {self.min_brightness:.2f})"
            )
        elif brightness > self.max_brightness:
            validation_info["warnings"].append(
                f"Image very bright: brightness {brightness:.2f} "
                f"(maximum: {self.max_brightness:.2f})"
            )
        
        # Contrast (standard deviation)
        contrast = np.std(img_array) / 255.0
        validation_info["metrics"]["contrast"] = contrast
        
        if contrast < self.min_contrast:
            validation_info["warnings"].append(
                f"Low contrast: {contrast:.2f} (minimum recommended: {self.min_contrast:.2f})"
            )
        
        # Sharpness (Laplacian variance)
        try:
            from cv2 import Laplacian, CV_64F
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            laplacian_var = Laplacian(gray, CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
            validation_info["metrics"]["sharpness"] = sharpness
            
            if sharpness < self.min_sharpness:
                validation_info["warnings"].append(
                    f"Image may be blurry: sharpness {sharpness:.2f} "
                    f"(minimum recommended: {self.min_sharpness:.2f})"
                )
        except ImportError:
            logger.warning("OpenCV not available, skipping sharpness check")
            validation_info["metrics"]["sharpness"] = None
        
        # Check for empty or corrupted image
        if img_array.size == 0:
            validation_info["valid"] = False
            validation_info["errors"].append("Image is empty or corrupted")
        
        # Overall validation
        validation_info["valid"] = len(validation_info["errors"]) == 0
        
        return validation_info["valid"], validation_info
    
    def get_validation_summary(self, validation_info: Dict[str, Any]) -> str:
        """
        Get human-readable validation summary.
        
        Args:
            validation_info: Validation info from validate()
            
        Returns:
            Summary string
        """
        if validation_info["valid"]:
            summary = "✓ Image validation passed"
        else:
            summary = "✗ Image validation failed"
        
        if validation_info["errors"]:
            summary += f"\n  Errors: {len(validation_info['errors'])}"
            for error in validation_info["errors"]:
                summary += f"\n    - {error}"
        
        if validation_info["warnings"]:
            summary += f"\n  Warnings: {len(validation_info['warnings'])}"
            for warning in validation_info["warnings"]:
                summary += f"\n    - {warning}"
        
        return summary
    
    # Static methods for backward compatibility with helpers
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



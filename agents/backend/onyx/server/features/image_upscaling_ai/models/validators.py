"""
Image Validators for Upscaling
================================

Comprehensive validation system for image upscaling operations.
"""

import logging
from typing import Tuple, Optional, Dict, Any, Union, List
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Validates images for upscaling operations.
    
    Features:
    - Resolution validation
    - Quality assessment
    - Format validation
    - Aspect ratio checks
    - Memory estimation
    """
    
    def __init__(
        self,
        min_resolution: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (8192, 8192),
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 10.0,
        min_brightness: float = 0.05,
        max_brightness: float = 0.98,
        min_contrast: float = 0.1,
        min_sharpness: float = 0.05,
        max_file_size_mb: float = 100.0,
    ):
        """
        Initialize image validator.
        
        Args:
            min_resolution: Minimum image resolution
            max_resolution: Maximum image resolution
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
            min_brightness: Minimum brightness (0.0-1.0)
            max_brightness: Maximum brightness (0.0-1.0)
            min_contrast: Minimum contrast
            min_sharpness: Minimum sharpness
            max_file_size_mb: Maximum file size in MB
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
        self.max_file_size_mb = max_file_size_mb
    
    def validate(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate image for upscaling.
        
        Args:
            image: Image to validate
            scale_factor: Optional scale factor to check output size
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }
        
        # Load image if path
        if isinstance(image, (str, Path)):
            try:
                file_size_mb = Path(image).stat().st_size / (1024 * 1024)
                if file_size_mb > self.max_file_size_mb:
                    validation_info["valid"] = False
                    validation_info["errors"].append(
                        f"File too large: {file_size_mb:.2f}MB (max: {self.max_file_size_mb}MB)"
                    )
                    return False, validation_info
                
                pil_image = Image.open(image)
            except Exception as e:
                validation_info["valid"] = False
                validation_info["errors"].append(f"Cannot open image: {e}")
                return False, validation_info
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            validation_info["valid"] = False
            validation_info["errors"].append(f"Unsupported image type: {type(image)}")
            return False, validation_info
        
        # Convert to RGB for analysis
        try:
            if pil_image.mode not in ["RGB", "RGBA", "L"]:
                pil_image = pil_image.convert("RGB")
                validation_info["warnings"].append(f"Converted from {pil_image.mode} to RGB")
        except Exception as e:
            validation_info["valid"] = False
            validation_info["errors"].append(f"Cannot convert image: {e}")
            return False, validation_info
        
        # Check resolution
        width, height = pil_image.size
        min_w, min_h = self.min_resolution
        max_w, max_h = self.max_resolution
        
        validation_info["metrics"]["resolution"] = (width, height)
        validation_info["metrics"]["pixels"] = width * height
        
        if width < min_w or height < min_h:
            validation_info["valid"] = False
            validation_info["errors"].append(
                f"Resolution too small: {width}x{height} (minimum: {min_w}x{min_h})"
            )
        elif width > max_w or height > max_h:
            validation_info["warnings"].append(
                f"Resolution very large: {width}x{height} (maximum: {max_w}x{max_h})"
            )
        
        # Check aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        validation_info["metrics"]["aspect_ratio"] = aspect_ratio
        
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            validation_info["valid"] = False
            validation_info["errors"].append(
                f"Aspect ratio out of range: {aspect_ratio:.2f} "
                f"(range: {self.min_aspect_ratio:.2f} to {self.max_aspect_ratio:.2f})"
            )
        
        # Check output size if scale factor provided
        if scale_factor:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            new_pixels = new_width * new_height
            
            validation_info["metrics"]["output_resolution"] = (new_width, new_height)
            validation_info["metrics"]["output_pixels"] = new_pixels
            
            # Estimate memory usage (rough: 4 bytes per pixel for RGB)
            estimated_mb = (new_pixels * 4) / (1024 * 1024)
            validation_info["metrics"]["estimated_memory_mb"] = estimated_mb
            
            if new_width > max_w or new_height > max_h:
                validation_info["warnings"].append(
                    f"Output resolution will be very large: {new_width}x{new_height}"
                )
            
            if estimated_mb > 500:  # Warn if > 500MB
                validation_info["warnings"].append(
                    f"High memory usage estimated: {estimated_mb:.1f}MB"
                )
        
        # Analyze image quality
        try:
            img_array = np.array(pil_image.convert("RGB"))
            
            # Brightness
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            brightness = np.mean(gray) / 255.0
            validation_info["metrics"]["brightness"] = brightness
            
            if brightness < self.min_brightness:
                validation_info["warnings"].append(
                    f"Image very dark: brightness {brightness:.2f}"
                )
            elif brightness > self.max_brightness:
                validation_info["warnings"].append(
                    f"Image very bright: brightness {brightness:.2f}"
                )
            
            # Contrast
            contrast = np.std(gray) / 255.0
            validation_info["metrics"]["contrast"] = contrast
            
            if contrast < self.min_contrast:
                validation_info["warnings"].append(
                    f"Low contrast: {contrast:.2f}"
                )
            
            # Sharpness (gradient-based)
            if CV2_AVAILABLE and len(gray.shape) == 2:
                laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
                sharpness = laplacian.var() / 10000.0  # Normalize
            else:
                grad_x = np.gradient(gray, axis=1)
                grad_y = np.gradient(gray, axis=0)
                sharpness = (np.var(grad_x) + np.var(grad_y)) / 10000.0
            
            validation_info["metrics"]["sharpness"] = sharpness
            
            if sharpness < self.min_sharpness:
                validation_info["warnings"].append(
                    f"Image may be blurry: sharpness {sharpness:.2f}"
                )
            
        except Exception as e:
            logger.warning(f"Error analyzing image quality: {e}")
            validation_info["warnings"].append(f"Could not analyze quality: {e}")
        
        return validation_info["valid"], validation_info


class ScaleFactorValidator:
    """Validates scale factors for upscaling."""
    
    @staticmethod
    def validate(
        scale_factor: float,
        min_scale: float = 1.0,
        max_scale: float = 8.0,
        original_size: Optional[Tuple[int, int]] = None,
        max_output_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate scale factor.
        
        Args:
            scale_factor: Scale factor to validate
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
            original_size: Original image size (optional)
            max_output_size: Maximum output size (optional)
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        if scale_factor < min_scale:
            errors.append(f"Scale factor {scale_factor} is below minimum {min_scale}")
        
        if scale_factor > max_scale:
            errors.append(f"Scale factor {scale_factor} exceeds maximum {max_scale}")
        
        # Check output size if provided
        if original_size and max_output_size:
            width, height = original_size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            max_w, max_h = max_output_size
            
            if new_width > max_w or new_height > max_h:
                errors.append(
                    f"Output size {new_width}x{new_height} exceeds maximum {max_w}x{max_h}"
                )
        
        return len(errors) == 0, errors


class MemoryEstimator:
    """Estimates memory usage for upscaling operations."""
    
    @staticmethod
    def estimate(
        image_size: Tuple[int, int],
        scale_factor: float,
        channels: int = 3,
        dtype_size: int = 4  # float32 = 4 bytes
    ) -> Dict[str, float]:
        """
        Estimate memory usage.
        
        Args:
            image_size: Original image size (width, height)
            scale_factor: Scale factor
            channels: Number of channels (3 for RGB)
            dtype_size: Size of data type in bytes
            
        Returns:
            Dictionary with memory estimates
        """
        width, height = image_size
        original_pixels = width * height
        new_pixels = int(original_pixels * (scale_factor ** 2))
        
        # Memory for original image
        original_mb = (original_pixels * channels * dtype_size) / (1024 * 1024)
        
        # Memory for upscaled image
        upscaled_mb = (new_pixels * channels * dtype_size) / (1024 * 1024)
        
        # Processing overhead (rough estimate: 2x for intermediate buffers)
        processing_mb = upscaled_mb * 2
        
        total_mb = original_mb + upscaled_mb + processing_mb
        
        return {
            "original_mb": round(original_mb, 2),
            "upscaled_mb": round(upscaled_mb, 2),
            "processing_mb": round(processing_mb, 2),
            "total_mb": round(total_mb, 2),
            "original_pixels": original_pixels,
            "upscaled_pixels": new_pixels,
        }



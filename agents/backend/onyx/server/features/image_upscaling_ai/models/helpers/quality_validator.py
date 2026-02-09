"""
Image Quality Validator
========================

Validates and enhances image quality before upscaling.
"""

import logging
import numpy as np
from typing import Union, Optional
from pathlib import Path
from PIL import Image, ImageEnhance

from .metrics import ImageQualityMetrics

logger = logging.getLogger(__name__)


class ImageQualityValidator:
    """Validates and assesses image quality before upscaling."""
    
    @staticmethod
    def validate_image(image: Union[Image.Image, str, Path, np.ndarray]) -> ImageQualityMetrics:
        """Validate image quality and return metrics."""
        warnings = []
        errors = []
        
        # Convert to PIL if needed
        pil_image = ImageQualityValidator._convert_to_pil(image)
        if pil_image is None:
            return ImageQualityMetrics(
                brightness=0.0, contrast=0.0, sharpness=0.0,
                resolution=(0, 0), is_valid=False,
                warnings=[], errors=[f"Unsupported image type: {type(image)}"]
            )
        
        width, height = pil_image.size
        resolution = (width, height)
        
        # Check minimum resolution
        if width < 32 or height < 32:
            errors.append(f"Image too small: {width}x{height}. Minimum: 32x32")
        
        # Convert to numpy for analysis
        img_array = np.array(pil_image)
        
        # Calculate metrics
        brightness = ImageQualityValidator._calculate_brightness(img_array)
        contrast = ImageQualityValidator._calculate_contrast(img_array)
        sharpness = ImageQualityValidator._calculate_sharpness(img_array)
        
        # Validate brightness
        if brightness < 10:
            errors.append(f"Image too dark: brightness {brightness:.1f}")
        elif brightness > 245:
            warnings.append(f"Image may be overexposed: brightness {brightness:.1f}")
        
        # Validate contrast
        if contrast < 5:
            warnings.append(f"Very low contrast: {contrast:.1f}")
        
        # Validate sharpness
        if sharpness < 50:
            warnings.append(f"Image may be blurry: sharpness {sharpness:.1f}")
        
        is_valid = len(errors) == 0
        
        return ImageQualityMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            resolution=resolution,
            is_valid=is_valid,
            warnings=warnings,
            errors=errors
        )
    
    @staticmethod
    def _convert_to_pil(image: Union[Image.Image, str, Path, np.ndarray]) -> Optional[Image.Image]:
        """Convert various image types to PIL Image."""
        try:
            if isinstance(image, (str, Path)):
                return Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                return image.convert("RGB")
            return None
        except Exception as e:
            logger.warning(f"Error converting image to PIL: {e}")
            return None
    
    @staticmethod
    def _calculate_brightness(img_array: np.ndarray) -> float:
        """Calculate image brightness."""
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        return float(np.mean(gray))
    
    @staticmethod
    def _calculate_contrast(img_array: np.ndarray) -> float:
        """Calculate image contrast."""
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        return float(np.std(gray))
    
    @staticmethod
    def _calculate_sharpness(img_array: np.ndarray) -> float:
        """Calculate image sharpness using gradient-based method."""
        try:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            grad_x = np.gradient(gray.astype(float), axis=1)
            grad_y = np.gradient(gray.astype(float), axis=0)
            return float(np.var(grad_x) + np.var(grad_y))
        except Exception:
            # Fallback to contrast-based estimate
            contrast = ImageQualityValidator._calculate_contrast(img_array)
            return contrast * 10
    
    @staticmethod
    def enhance_image(image: Image.Image, strength: float = 1.0) -> Image.Image:
        """Enhance image quality if needed."""
        if strength <= 0:
            return image
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.0 + (strength * 0.1))
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.0 + (strength * 0.05))
        
        return image


"""
Image Analyzer Helper
=====================

Analyzes image properties and characteristics.
"""

import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes image properties and characteristics."""
    
    @staticmethod
    def get_brightness(image: Image.Image) -> float:
        """
        Calculate image brightness.
        
        Args:
            image: PIL Image
            
        Returns:
            Brightness value (0.0 - 1.0)
        """
        img_array = np.array(image.convert("L"))
        return float(img_array.mean() / 255.0)
    
    @staticmethod
    def get_contrast(image: Image.Image) -> float:
        """
        Calculate image contrast.
        
        Args:
            image: PIL Image
            
        Returns:
            Contrast value (standard deviation)
        """
        img_array = np.array(image.convert("L"))
        return float(img_array.std())
    
    @staticmethod
    def get_sharpness(image: Image.Image) -> float:
        """
        Calculate image sharpness using variance method.
        
        Args:
            image: PIL Image
            
        Returns:
            Sharpness value
        """
        try:
            from scipy import ndimage
            img_array = np.array(image.convert("L"))
            laplacian = ndimage.laplacian(img_array)
            return float(laplacian.var())
        except ImportError:
            # Fallback: simple edge detection
            from PIL import ImageFilter
            edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            return float(edges_array.std())
    
    @staticmethod
    def analyze_quality(image: Image.Image) -> dict:
        """
        Analyze overall image quality.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with quality metrics
        """
        brightness = ImageAnalyzer.get_brightness(image)
        contrast = ImageAnalyzer.get_contrast(image)
        
        # Normalize contrast
        contrast_normalized = min(contrast / 50.0, 1.0)
        
        # Calculate quality score
        quality_score = (brightness * 0.3 + contrast_normalized * 0.7)
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "quality_score": quality_score,
            "size": image.size,
            "mode": image.mode,
        }
    
    @staticmethod
    def needs_enhancement(image: Image.Image) -> dict:
        """
        Determine if image needs enhancement.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with enhancement recommendations
        """
        brightness = ImageAnalyzer.get_brightness(image)
        contrast = ImageAnalyzer.get_contrast(image)
        
        needs_contrast = contrast < 30
        needs_brightness = brightness < 0.3 or brightness > 0.8
        needs_sharpness = True  # Always beneficial
        
        return {
            "needs_contrast": needs_contrast,
            "needs_brightness": needs_brightness,
            "needs_sharpness": needs_sharpness,
            "recommendations": {
                "enhance_contrast": needs_contrast,
                "enhance_brightness": needs_brightness,
                "enhance_sharpness": needs_sharpness,
            }
        }


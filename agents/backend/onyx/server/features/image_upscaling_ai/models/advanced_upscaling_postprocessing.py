"""
Advanced Upscaling Postprocessing
==================================

Postprocessing methods for upscaled images.
"""

import logging
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .helpers import FilterApplicator


class PostprocessingMethods:
    """Postprocessing methods for upscaled images."""
    
    @staticmethod
    def apply_anti_aliasing(
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """Apply anti-aliasing to reduce pixelation artifacts."""
        if strength <= 0:
            return image
        
        if not CV2_AVAILABLE:
            return image.filter(ImageFilter.SMOOTH_MORE)
        
        try:
            img_array = np.array(image, dtype=np.float32)
            
            if len(img_array.shape) == 3:
                blurred = np.zeros_like(img_array)
                for i in range(3):
                    blurred[:, :, i] = cv2.GaussianBlur(
                        img_array[:, :, i], (0, 0), sigmaX=strength * 0.5
                    )
            else:
                blurred = cv2.GaussianBlur(img_array, (0, 0), sigmaX=strength * 0.5)
            
            result = (1 - strength * 0.3) * img_array + (strength * 0.3) * blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            return Image.fromarray(result)
            
        except Exception as e:
            logger.warning(f"Anti-aliasing failed: {e}")
            return image
    
    @staticmethod
    def reduce_artifacts(
        image: Image.Image,
        method: str = "bilateral"
    ) -> Image.Image:
        """Reduce upscaling artifacts."""
        if not CV2_AVAILABLE:
            return image.filter(ImageFilter.SMOOTH)
        
        try:
            img_array = np.array(image)
            
            if method == "bilateral":
                filtered = cv2.bilateralFilter(
                    img_array, d=9, sigmaColor=75, sigmaSpace=75
                )
            elif method == "median":
                filtered = cv2.medianBlur(img_array, 5)
            elif method == "gaussian":
                filtered = cv2.GaussianBlur(img_array, (5, 5), 0)
            else:
                filtered = img_array
            
            return Image.fromarray(filtered)
            
        except Exception as e:
            logger.warning(f"Artifact reduction failed: {e}")
            return image
    
    @staticmethod
    def enhance_edges(
        image: Image.Image,
        strength: float = 1.2
    ) -> Image.Image:
        """Enhance edges in the image."""
        return FilterApplicator.apply_unsharp_mask(
            image, radius=2, percent=int(100 * strength), threshold=3
        )



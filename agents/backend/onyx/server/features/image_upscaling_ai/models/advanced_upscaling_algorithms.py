"""
Advanced Upscaling Algorithms
=============================

Static methods for upscaling algorithms.
"""

import logging
import numpy as np
from typing import Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from typing import Optional
from .helpers import (
    DimensionCalculator,
    FilterApplicator,
    UpscalingAlgorithms,
    ImageProcessingUtils,
)


class UpscalingAlgorithmsStatic:
    """Static methods for upscaling algorithms."""
    
    @staticmethod
    def upscale_lanczos(
        image: Image.Image,
        scale_factor: float,
        taps: int = 3
    ) -> Image.Image:
        """Upscale using Lanczos resampling."""
        width, height = DimensionCalculator.get_image_dimensions(image)
        new_width, new_height = DimensionCalculator.calculate_new_dimensions(
            width, height, scale_factor
        )
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def upscale_bicubic_enhanced(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Enhanced bicubic upscaling with post-processing."""
        width, height = DimensionCalculator.get_image_dimensions(image)
        new_width, new_height = DimensionCalculator.calculate_new_dimensions(
            width, height, scale_factor
        )
        upscaled = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        upscaled = FilterApplicator.apply_unsharp_mask(
            upscaled, radius=1, percent=150, threshold=3
        )
        return upscaled
    
    @staticmethod
    def upscale_opencv_edsr(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Upscale using OpenCV's EDSR-like super resolution."""
        return UpscalingAlgorithms.upscale_opencv_edsr(image, scale_factor)
    
    @staticmethod
    def multi_scale_upscale(
        image: Image.Image,
        scale_factor: float,
        passes: int = 2
    ) -> Image.Image:
        """Multi-scale upscaling for better quality."""
        return UpscalingAlgorithms.multi_scale_upscale(image, scale_factor, passes)
    
    @staticmethod
    def upscale_adaptive(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Adaptive upscaling based on image characteristics."""
        return UpscalingAlgorithms.upscale_adaptive(image, scale_factor)
    
    @staticmethod
    def upscale_esrgan_like(
        image: Image.Image,
        scale_factor: float,
        iterations: int = 2
    ) -> Image.Image:
        """ESRGAN-like upscaling using iterative enhancement."""
        return UpscalingAlgorithms.upscale_esrgan_like(image, scale_factor, iterations)
    
    @staticmethod
    def upscale_waifu2x_like(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Waifu2x-like upscaling with noise reduction."""
        return UpscalingAlgorithms.upscale_waifu2x_like(image, scale_factor)
    
    @staticmethod
    def upscale_real_esrgan_like(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Real-ESRGAN-like upscaling with advanced processing."""
        return UpscalingAlgorithms.upscale_real_esrgan_like(image, scale_factor)
    
    @staticmethod
    def upscale_realesrgan(
        image: Image.Image,
        scale_factor: float,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> Image.Image:
        """Upscale using Real-ESRGAN model."""
        try:
            from .realesrgan_integration import RealESRGANUpscaler, REALESRGAN_AVAILABLE
            
            if not REALESRGAN_AVAILABLE:
                logger.warning("Real-ESRGAN not available, falling back to Lanczos")
                return UpscalingAlgorithmsStatic.upscale_lanczos(image, scale_factor)
            
            upscaler = RealESRGANUpscaler(model_name=model_name, device=device)
            return upscaler.upscale(image, scale_factor)
            
        except Exception as e:
            logger.warning(f"Real-ESRGAN upscaling failed: {e}, falling back to Lanczos")
            return UpscalingAlgorithmsStatic.upscale_lanczos(image, scale_factor)


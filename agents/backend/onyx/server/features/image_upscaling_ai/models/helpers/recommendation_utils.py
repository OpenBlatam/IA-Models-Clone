"""
Recommendation Utilities
========================

Utilities for recommending upscaling methods based on image characteristics.
"""

import logging
from typing import Union, Callable
from pathlib import Path
from PIL import Image

from .quality_calculator_utils import QualityCalculator
from .method_selector_utils import MethodSelector

logger = logging.getLogger(__name__)


class RecommendationUtils:
    """Utilities for recommending upscaling methods."""
    
    @staticmethod
    def get_recommended_method(
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        priority: str = "quality"
    ) -> str:
        """
        Get recommended upscaling method based on image and requirements.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            priority: Priority ('quality', 'speed', 'balanced')
            
        Returns:
            Recommended method name
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        quality = QualityCalculator.calculate_quality_metrics(pil_image)
        width, height = pil_image.size
        
        # Use MethodSelector for balanced priority, otherwise use custom logic
        if priority == "balanced":
            return MethodSelector.select_best_method(pil_image, scale_factor, quality)
        
        # Custom logic for quality and speed priorities
        if priority == "quality":
            if scale_factor > 4.0:
                return "multi_scale"
            elif quality.noise_level > 15:
                return "real_esrgan_like"
            elif quality.overall_quality < 0.6:
                return "esrgan_like"
            else:
                return "bicubic"
        else:  # speed
            if width < 512 and height < 512:
                return "lanczos"
            else:
                return "bicubic"
    
    @staticmethod
    def upscale_smart(
        upscale_func: Callable,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        priority: str = "quality",
        return_metrics: bool = False
    ) -> Union[Image.Image, tuple]:
        """
        Smart upscaling with automatic method selection.
        
        Args:
            upscale_func: Function to upscale (image, scale_factor, method, return_metrics) -> result
            image: Input image
            scale_factor: Scale factor
            priority: Priority ('quality', 'speed', 'balanced')
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        method = RecommendationUtils.get_recommended_method(image, scale_factor, priority)
        return upscale_func(image, scale_factor, method, return_metrics=return_metrics)


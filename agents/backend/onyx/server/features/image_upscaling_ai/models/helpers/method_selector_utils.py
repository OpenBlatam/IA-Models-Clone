"""
Method Selector Utilities
=========================

Utilities for automatically selecting the best upscaling method.
"""

import logging
from PIL import Image

from .quality_calculator_utils import QualityCalculator

logger = logging.getLogger(__name__)


class MethodSelector:
    """Selector for best upscaling method based on image characteristics."""
    
    @staticmethod
    def select_best_method(
        image: Image.Image,
        scale_factor: float,
        auto_select: bool = False,
        stats: dict = None
    ) -> str:
        """
        Automatically select best upscaling method based on image characteristics.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            auto_select: Whether to auto-select method
            stats: Statistics dictionary to update
            
        Returns:
            Best method name
        """
        if not auto_select:
            return "lanczos"
        
        # Analyze image
        quality = QualityCalculator.calculate_quality_metrics(image)
        width, height = image.size
        
        # Select method based on characteristics
        if scale_factor > 4.0:
            # Very large scale - use multi_scale
            method = "multi_scale"
        elif quality.noise_level > 20:
            # High noise - use opencv with denoising
            method = "opencv"
        elif quality.sharpness < 200:
            # Low sharpness - use bicubic with enhancement
            method = "bicubic"
        elif width < 256 or height < 256:
            # Small image - use lanczos (fast and good quality)
            method = "lanczos"
        else:
            # Default to lanczos
            method = "lanczos"
        
        # Track method selection
        if stats is not None:
            stats["method_selections"] = stats.get("method_selections", {})
            stats["method_selections"][method] = stats["method_selections"].get(method, 0) + 1
        
        return method



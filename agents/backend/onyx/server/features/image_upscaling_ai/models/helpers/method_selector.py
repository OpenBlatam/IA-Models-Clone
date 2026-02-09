"""
Method Selector
===============

Automatically select the best upscaling method based on image characteristics.
"""

import logging
from typing import Dict, Any
from PIL import Image

from .quality_calculator import QualityCalculator

logger = logging.getLogger(__name__)


class MethodSelector:
    """Select best upscaling method based on image characteristics."""
    
    @staticmethod
    def select_best_method(
        image: Image.Image,
        scale_factor: float,
        quality_metrics: Any = None
    ) -> str:
        """
        Automatically select best upscaling method based on image characteristics.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            quality_metrics: Optional pre-calculated quality metrics
            
        Returns:
            Best method name
        """
        # Calculate quality metrics if not provided
        if quality_metrics is None:
            quality_metrics = QualityCalculator.calculate_quality_metrics(image)
        
        width, height = image.size
        
        # Select method based on characteristics
        if scale_factor > 4.0:
            # Very large scale - use multi_scale
            return "multi_scale"
        elif quality_metrics.noise_level > 20:
            # High noise - use opencv with denoising
            return "opencv"
        elif quality_metrics.sharpness < 200:
            # Low sharpness - use bicubic with enhancement
            return "bicubic"
        elif width < 256 or height < 256:
            # Small image - use lanczos (fast and good quality)
            return "lanczos"
        else:
            # Default to lanczos
            return "lanczos"
    
    @staticmethod
    def get_method_recommendation(
        image: Image.Image,
        scale_factor: float,
        quality_metrics: Any = None
    ) -> Dict[str, Any]:
        """
        Get detailed method recommendation with reasoning.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            quality_metrics: Optional pre-calculated quality metrics
            
        Returns:
            Dictionary with method recommendation and reasoning
        """
        if quality_metrics is None:
            quality_metrics = QualityCalculator.calculate_quality_metrics(image)
        
        method = MethodSelector.select_best_method(image, scale_factor, quality_metrics)
        
        reasoning = []
        if scale_factor > 4.0:
            reasoning.append(f"Large scale factor ({scale_factor}x) requires multi-scale approach")
        elif quality_metrics.noise_level > 20:
            reasoning.append(f"High noise level ({quality_metrics.noise_level:.1f}) requires denoising")
        elif quality_metrics.sharpness < 200:
            reasoning.append(f"Low sharpness ({quality_metrics.sharpness:.1f}) requires enhancement")
        else:
            reasoning.append("Standard upscaling method suitable")
        
        return {
            "method": method,
            "reasoning": reasoning,
            "quality_metrics": {
                "sharpness": quality_metrics.sharpness,
                "noise_level": quality_metrics.noise_level,
                "overall_quality": quality_metrics.overall_quality,
            }
        }



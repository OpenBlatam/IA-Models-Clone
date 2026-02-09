"""
Analysis Mixin

Contains image analysis and reporting methods.
"""

import logging
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from PIL import Image

from ..helpers import (
    ImageAnalysisUtils,
    MethodSelector,
    MethodComparisonUtils,
    QualityCalculator,
)

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """
    Mixin providing analysis and reporting functionality.
    
    This mixin contains:
    - Image characteristics analysis
    - Processing recommendations
    - Method comparison
    - Performance benchmarking
    - Comprehensive reporting
    """
    
    def analyze_image_characteristics(
        self,
        image: Union[Image.Image, str, Path]
    ) -> Dict[str, Any]:
        """Analyze image characteristics."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        return ImageAnalysisUtils.analyze_image_characteristics(pil_image)
    
    def get_processing_recommendations(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float
    ) -> Dict[str, Any]:
        """Get processing recommendations for an image."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
        recommended_method = MethodSelector.select_best_method(pil_image, scale_factor)
        
        return {
            "recommended_method": recommended_method,
            "analysis": analysis,
            "processing_options": {
                "use_frequency_analysis": analysis.get("quality_metrics", {}).get("sharpness", 0) < 500,
                "use_adaptive_contrast": analysis.get("quality_metrics", {}).get("contrast", 0) < 30,
                "use_texture_enhancement": analysis.get("edge_analysis", {}).get("edge_density", 0) > 0.1,
                "use_color_enhancement": True,
                "use_regional_processing": pil_image.size[0] * pil_image.size[1] > 2000000,
            },
            "estimated_time": 1.0,
            "quality_expectation": 0.85,
        }
    
    def compare_methods(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different upscaling methods."""
        return MethodComparisonUtils.compare_methods(
            self.upscale,
            image,
            scale_factor,
            methods=methods,
            use_cache=False
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not hasattr(self, 'stats'):
            return {}
        
        return {
            "upscales_performed": self.stats.get("upscales_performed", 0),
            "successful_upscales": self.stats.get("successful_upscales", 0),
            "failed_upscales": self.stats.get("failed_upscales", 0),
            "cache_hits": self.stats.get("cache_hits", 0),
            "cache_misses": self.stats.get("cache_misses", 0),
            "total_time": self.stats.get("total_time", 0.0),
            "avg_time": (
                self.stats.get("total_time", 0.0) / self.stats.get("upscales_performed", 1)
                if self.stats.get("upscales_performed", 0) > 0 else 0.0
            ),
        }



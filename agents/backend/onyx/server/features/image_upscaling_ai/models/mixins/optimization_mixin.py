"""
Optimization Mixin

Contains optimization and performance enhancement methods.
"""

import logging
import time
from typing import Union, Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityCalculator,
    OptimizationUtils,
    MethodSelector,
)

logger = logging.getLogger(__name__)


class OptimizationMixin:
    """
    Mixin providing optimization functionality.
    
    This mixin contains:
    - Performance optimization
    - Quality optimization
    - Method optimization
    - Resource optimization
    - Adaptive optimization
    """
    
    def optimize_upscaling_method(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        target_quality: float = 0.85,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize upscaling method for best quality/speed ratio.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            target_quality: Target quality score
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        methods = ["lanczos", "bicubic", "opencv", "multi_scale", "real_esrgan_like"]
        best_method = None
        best_quality = 0.0
        best_time = float('inf')
        results = {}
        
        for method in methods:
            start_time = time.time()
            try:
                result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
                processing_time = time.time() - start_time
                quality = QualityCalculator.calculate_quality_metrics(result)
                
                results[method] = {
                    "quality": quality.overall_quality,
                    "time": processing_time,
                    "score": quality.overall_quality / processing_time if processing_time > 0 else 0
                }
                
                if quality.overall_quality > best_quality:
                    best_quality = quality.overall_quality
                    best_method = method
                
                if processing_time < best_time:
                    best_time = processing_time
                
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                results[method] = {"error": str(e)}
        
        return {
            "best_method": best_method,
            "best_quality": best_quality,
            "best_time": best_time,
            "results": results,
            "recommendation": best_method if best_method else "lanczos"
        }
    
    def optimize_for_speed(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        min_quality: float = 0.7
    ) -> Image.Image:
        """
        Optimize upscaling for maximum speed while maintaining minimum quality.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            min_quality: Minimum acceptable quality
            
        Returns:
            Upscaled image
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        # Fast methods in order of speed
        fast_methods = ["lanczos", "bicubic", "opencv"]
        
        for method in fast_methods:
            result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
            quality = QualityCalculator.calculate_quality_metrics(result)
            
            if quality.overall_quality >= min_quality:
                logger.info(f"Selected {method} for speed optimization (quality: {quality.overall_quality:.3f})")
                return result
        
        # If no fast method meets quality, use best available
        logger.warning(f"No fast method met quality threshold, using best available")
        return self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
    
    def optimize_for_quality(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        max_time: Optional[float] = None
    ) -> Image.Image:
        """
        Optimize upscaling for maximum quality.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            max_time: Maximum processing time in seconds
            
        Returns:
            Upscaled image
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        # Quality methods in order of quality
        quality_methods = ["real_esrgan_like", "esrgan_like", "waifu2x_like", "multi_scale"]
        
        for method in quality_methods:
            start_time = time.time()
            result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
            processing_time = time.time() - start_time
            
            if max_time is None or processing_time <= max_time:
                quality = QualityCalculator.calculate_quality_metrics(result)
                logger.info(f"Selected {method} for quality optimization (quality: {quality.overall_quality:.3f})")
                return result
        
        # Fallback to best available
        return self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
    
    def get_optimization_recommendations(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        priority: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations based on image and priority.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            priority: Priority ('speed', 'quality', 'balanced')
            
        Returns:
            Dictionary with recommendations
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        analysis = self.analyze_image_characteristics(pil_image)
        recommended_method = MethodSelector.select_best_method(pil_image, scale_factor)
        
        if priority == "speed":
            recommended_methods = ["lanczos", "bicubic", "opencv"]
            recommended_method = recommended_methods[0] if recommended_method in recommended_methods else recommended_methods[0]
        elif priority == "quality":
            recommended_methods = ["real_esrgan_like", "esrgan_like", "waifu2x_like"]
            recommended_method = recommended_methods[0] if recommended_method in recommended_methods else recommended_methods[0]
        
        return {
            "recommended_method": recommended_method,
            "priority": priority,
            "analysis": analysis,
            "estimated_time": 1.0 if priority == "speed" else 5.0,
            "estimated_quality": 0.75 if priority == "speed" else 0.90,
            "optimization_tips": [
                "Use caching for repeated upscales",
                "Enable auto method selection",
                "Use batch processing for multiple images"
            ]
        }



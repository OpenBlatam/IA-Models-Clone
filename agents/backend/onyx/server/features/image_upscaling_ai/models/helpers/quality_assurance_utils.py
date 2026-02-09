"""
Quality Assurance Utilities
===========================

Utilities for quality checking and retry logic with different methods.
"""

import logging
from typing import Tuple, List, Callable, Union, Optional
from pathlib import Path
from PIL import Image

from .metrics_utils import UpscalingMetrics
from .quality_calculator_utils import QualityCalculator

logger = logging.getLogger(__name__)


class QualityAssuranceUtils:
    """Utilities for quality assurance and retry logic."""
    
    @staticmethod
    def upscale_with_quality_check(
        upscale_func: Callable,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        min_quality_threshold: float = 0.6,
        max_attempts: int = 3,
        fallback_methods: Optional[List[str]] = None,
        **upscale_kwargs
    ) -> Tuple[Image.Image, UpscalingMetrics]:
        """
        Upscale with quality check and automatic retry with different methods.
        
        Args:
            upscale_func: Function to upscale (image, scale_factor, method, return_metrics=True) -> (image, metrics)
            image: Input image
            scale_factor: Scale factor
            method: Initial method to try
            min_quality_threshold: Minimum quality threshold
            max_attempts: Maximum attempts with different methods
            fallback_methods: List of fallback methods (default: ["bicubic", "opencv", "multi_scale", "adaptive"])
            **upscale_kwargs: Additional arguments for upscale_func
            
        Returns:
            Tuple of (upscaled image, metrics)
        """
        if fallback_methods is None:
            fallback_methods = ["bicubic", "opencv", "multi_scale", "adaptive"]
        
        methods_to_try = [method] + fallback_methods
        best_result = None
        best_metrics = None
        best_quality = 0.0
        
        for attempt, current_method in enumerate(methods_to_try[:max_attempts]):
            try:
                result = upscale_func(
                    image,
                    scale_factor,
                    current_method,
                    return_metrics=True,
                    **upscale_kwargs
                )
                
                if isinstance(result, tuple):
                    upscaled_image, metrics = result
                else:
                    # If function doesn't return metrics, calculate them
                    if isinstance(image, (str, Path)):
                        pil_image = Image.open(image).convert("RGB")
                    else:
                        pil_image = image
                    quality = QualityCalculator.calculate_quality_metrics(result)
                    metrics = UpscalingMetrics(
                        original_size=pil_image.size,
                        upscaled_size=result.size,
                        scale_factor=scale_factor,
                        processing_time=0.0,
                        quality_score=quality.overall_quality,
                        sharpness_score=quality.sharpness,
                        artifact_score=1.0 - quality.artifact_count,
                        method_used=current_method,
                        success=True,
                    )
                    upscaled_image = result
                
                if metrics.quality_score and metrics.quality_score >= min_quality_threshold:
                    # Quality is acceptable
                    return upscaled_image, metrics
                
                # Track best result so far
                if metrics.quality_score and metrics.quality_score > best_quality:
                    best_quality = metrics.quality_score
                    best_result = upscaled_image
                    best_metrics = metrics
                    if hasattr(best_metrics, 'warnings'):
                        best_metrics.warnings.append(
                            f"Quality below threshold, tried {current_method}"
                        )
                    
            except Exception as e:
                logger.warning(f"Method {current_method} failed: {e}")
                continue
        
        # Return best result or raise error
        if best_result is None:
            raise RuntimeError(f"All upscaling methods failed for image")
        
        if best_metrics.quality_score and best_metrics.quality_score < min_quality_threshold:
            if hasattr(best_metrics, 'warnings'):
                best_metrics.warnings.append(
                    f"Best quality {best_quality:.3f} below threshold {min_quality_threshold}"
                )
        
        return best_result, best_metrics



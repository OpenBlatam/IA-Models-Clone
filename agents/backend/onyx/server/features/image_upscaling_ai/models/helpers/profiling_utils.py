"""
Profiling Utilities
==================

Utilities for profiling upscaling performance.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Callable, Union
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ProfilingUtils:
    """Utilities for profiling upscaling operations."""
    
    @staticmethod
    def profile_upscale(
        upscale_func: Callable,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        iterations: int = 5,
        **upscale_kwargs
    ) -> Dict[str, Any]:
        """
        Profile upscaling performance with multiple iterations.
        
        Args:
            upscale_func: Function to profile (image, scale_factor, method, return_metrics=True) -> (image, metrics)
            image: Input image
            scale_factor: Scale factor
            method: Method to profile
            iterations: Number of iterations
            **upscale_kwargs: Additional arguments for upscale_func
            
        Returns:
            Performance profile dictionary
        """
        times = []
        quality_scores = []
        
        for i in range(iterations):
            start_time = time.time()
            result = upscale_func(
                image,
                scale_factor,
                method,
                use_cache=False,  # Disable cache for accurate profiling
                return_metrics=True,
                **upscale_kwargs
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if isinstance(result, tuple):
                _, metrics = result
                if hasattr(metrics, 'quality_score') and metrics.quality_score:
                    quality_scores.append(metrics.quality_score)
            elif hasattr(result, 'quality_score') and result.quality_score:
                quality_scores.append(result.quality_score)
        
        return {
            "method": method,
            "iterations": iterations,
            "avg_time": sum(times) / len(times) if times else 0.0,
            "min_time": min(times) if times else 0.0,
            "max_time": max(times) if times else 0.0,
            "std_time": float(np.std(times)) if times else 0.0,
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else None,
            "times": times,
            "quality_scores": quality_scores,
        }



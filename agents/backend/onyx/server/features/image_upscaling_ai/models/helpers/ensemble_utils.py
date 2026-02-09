"""
Ensemble Utilities
==================

Utilities for ensemble methods and fusion of multiple upscaling results.
"""

import logging
import numpy as np
import time
from typing import List, Tuple, Optional, Callable, Union
from PIL import Image

from .metrics_utils import UpscalingMetrics
from .quality_calculator_utils import QualityCalculator

logger = logging.getLogger(__name__)


class EnsembleUtils:
    """Utilities for ensemble methods and fusion."""
    
    @staticmethod
    def fuse_results(
        results: List[Tuple[Image.Image, float]],
        fusion_method: str = "weighted_average"
    ) -> Image.Image:
        """
        Fuse multiple upscaling results.
        
        Args:
            results: List of (image, quality_score) tuples
            fusion_method: Fusion method ('weighted_average', 'best_quality', 'median')
            
        Returns:
            Fused image
        """
        if not results:
            raise ValueError("No results to fuse")
        
        if fusion_method == "weighted_average":
            # Weight by quality
            total_quality = sum(q for _, q in results)
            if total_quality > 0:
                weights = [q / total_quality for _, q in results]
            else:
                weights = [1.0 / len(results)] * len(results)
            
            # Weighted average
            result_arrays = [np.array(img, dtype=np.float32) for img, _ in results]
            fused_array = np.zeros_like(result_arrays[0])
            for arr, weight in zip(result_arrays, weights):
                fused_array += arr * weight
            fused_array = np.clip(fused_array, 0, 255).astype(np.uint8)
            return Image.fromarray(fused_array)
            
        elif fusion_method == "best_quality":
            # Use best quality result
            qualities = [q for _, q in results]
            best_idx = qualities.index(max(qualities))
            return results[best_idx][0]
            
        elif fusion_method == "median":
            # Median fusion
            result_arrays = [np.array(img) for img, _ in results]
            fused_array = np.median(result_arrays, axis=0).astype(np.uint8)
            return Image.fromarray(fused_array)
            
        else:
            # Default to weighted average
            total_quality = sum(q for _, q in results)
            weights = [q / total_quality if total_quality > 0 else 1.0 / len(results) for _, q in results]
            result_arrays = [np.array(img, dtype=np.float32) for img, _ in results]
            fused_array = np.zeros_like(result_arrays[0])
            for arr, weight in zip(result_arrays, weights):
                fused_array += arr * weight
            fused_array = np.clip(fused_array, 0, 255).astype(np.uint8)
            return Image.fromarray(fused_array)
    
    @staticmethod
    def create_ensemble(
        upscale_func: Callable,
        image: Image.Image,
        scale_factor: float,
        methods: List[str],
        fusion_method: str = "weighted_average"
    ) -> Tuple[Image.Image, List[Tuple[Image.Image, float]]]:
        """
        Create ensemble by upscaling with multiple methods.
        
        Args:
            upscale_func: Function to upscale (image, scale_factor, method) -> Image
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to use
            fusion_method: Fusion method
            
        Returns:
            Tuple of (fused_image, results_list)
        """
        results = []
        qualities = []
        
        for method in methods:
            try:
                result = upscale_func(image, scale_factor, method)
                quality = QualityCalculator.calculate_quality_metrics(result)
                results.append((result, quality.overall_quality))
                qualities.append(quality.overall_quality)
            except Exception as e:
                logger.warning(f"Method {method} failed in ensemble: {e}")
                continue
        
        if not results:
            raise RuntimeError("All methods failed in ensemble")
        
        fused = EnsembleUtils.fuse_results(results, fusion_method)
        return fused, results



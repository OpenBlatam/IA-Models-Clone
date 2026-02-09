"""
Advanced Ensemble and Optimization Methods
=========================================

Ensemble methods and advanced optimization techniques.
"""

import logging
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from PIL import Image

from .helpers import (
    UpscalingMetrics,
    EnsembleUtils,
    OptimizationUtils,
)

logger = logging.getLogger(__name__)


class EnsembleMethods:
    """Ensemble and optimization methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
    
    def upscale_with_ensemble(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        fusion_method: str = "weighted_average",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale using ensemble of multiple methods."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        if methods is None:
            methods = ["lanczos", "bicubic", "opencv"]
        
        def upscale_func(img, sf, method):
            return self.base_upscaler.upscale(img, sf, method, return_metrics=False)
        
        fused_result, results = EnsembleUtils.create_ensemble(
            upscale_func,
            pil_image,
            scale_factor,
            methods,
            fusion_method
        )
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(fused_result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=fused_result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"ensemble_{fusion_method}",
            success=True,
        )
        
        if return_metrics:
            return fused_result, metrics
        return fused_result
    
    def upscale_with_multi_scale_ensemble(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        ensemble_size: int = 3,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with multi-scale ensemble."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Create ensemble with different scale factors
        results = []
        scales = np.linspace(scale_factor * 0.9, scale_factor * 1.1, ensemble_size)
        
        for scale in scales:
            result = self.base_upscaler.upscale(pil_image, scale, "lanczos", return_metrics=False)
            # Resize to target size
            target_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            result = result.resize(target_size, Image.Resampling.LANCZOS)
            results.append(result)
        
        # Fuse results
        fused = EnsembleUtils.weighted_fusion(results, weights=None)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(fused)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=fused.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used="multi_scale_ensemble",
            success=True,
        )
        
        if return_metrics:
            return fused, metrics
        return fused
    
    def upscale_with_intelligent_fusion(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        fusion_methods: List[str] = None,
        fusion_weights: List[float] = None,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with intelligent fusion using learned weights."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        if fusion_methods is None:
            fusion_methods = ["lanczos", "opencv", "esrgan_like", "real_esrgan_like"]
        
        # Upscale with each method
        results = []
        qualities = []
        
        for method in fusion_methods:
            try:
                result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
                quality = self.base_upscaler.calculate_quality_metrics(result)
                results.append((result, quality.overall_quality))
                qualities.append(quality.overall_quality)
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        if not results:
            raise RuntimeError("All methods failed")
        
        # Intelligent weight calculation
        if fusion_weights is None:
            weights = EnsembleUtils.calculate_intelligent_weights(
                results,
                pil_image,
                self.base_upscaler
            )
        else:
            weights = fusion_weights
            if len(weights) != len(results):
                weights = weights[:len(results)]
                total = sum(weights)
                weights = [w / total if total > 0 else 1.0 / len(weights) for w in weights]
        
        # Intelligent fusion
        result_arrays = [np.array(img, dtype=np.float32) for img, _ in results]
        fused = np.zeros_like(result_arrays[0])
        
        for arr, weight in zip(result_arrays, weights):
            fused += arr * weight
        
        result = Image.fromarray(fused.astype(np.uint8))
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used="intelligent_fusion",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_ensemble_learning(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        learning_iterations: int = 5,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with ensemble learning approach."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Initial ensemble
        methods = ["lanczos", "bicubic", "opencv"]
        results = []
        
        for method in methods:
            result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
            results.append(result)
        
        # Learning iterations
        for iteration in range(learning_iterations):
            # Evaluate each result
            qualities = [self.base_upscaler.calculate_quality_metrics(r).overall_quality for r in results]
            
            # Update weights based on quality
            weights = [q / sum(qualities) for q in qualities]
            
            # Fuse with updated weights
            fused = EnsembleUtils.weighted_fusion(results, weights)
            
            # Refine fused result
            fused = self.base_upscaler.enhance_edges(fused, strength=1.1)
            
            # Replace worst result with fused
            worst_idx = qualities.index(min(qualities))
            results[worst_idx] = fused
        
        # Final fusion
        final = EnsembleUtils.weighted_fusion(results, weights=None)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(final)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=final.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"ensemble_learning_{learning_iterations}",
            success=True,
        )
        
        if return_metrics:
            return final, metrics
        return final



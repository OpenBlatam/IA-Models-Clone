"""
Advanced Methods Mixin

Contains advanced upscaling methods and techniques.
"""

import logging
import time
import numpy as np
from typing import Union, Tuple, Optional, List, Dict, Any
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityCalculator,
    PostprocessingMethods,
    ImageAnalysisUtils,
    MethodSelector,
)

logger = logging.getLogger(__name__)


class AdvancedMethodsMixin:
    """
    Mixin providing advanced upscaling methods.
    
    This mixin contains:
    - Smart enhancement
    - Quality boosting
    - Hybrid methods
    - Adaptive quality control
    - Progressive enhancement
    - Multi-scale fusion
    """
    
    def upscale_with_smart_enhancement(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        enhancement_mode: str = "auto",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale with smart enhancement that adapts to image content.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            enhancement_mode: Enhancement mode ('auto', 'portrait', 'landscape', 'text', 'art')
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Analyze image
        analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
        
        # Auto-detect mode if needed
        if enhancement_mode == "auto":
            edge_density = analysis.get("edge_analysis", {}).get("edge_density", 0)
            quality = analysis.get("quality_metrics", {}).get("overall_quality", 0.5)
            
            if edge_density > 0.2:
                enhancement_mode = "text"
            elif quality > 0.7:
                enhancement_mode = "art"
            elif pil_image.size[0] > pil_image.size[1]:
                enhancement_mode = "landscape"
            else:
                enhancement_mode = "portrait"
        
        # Upscale base
        result = self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        # Apply mode-specific enhancements
        if enhancement_mode == "portrait":
            result = PostprocessingMethods.enhance_edges(result, strength=1.2)
            result = PostprocessingMethods.color_enhancement(result, saturation=1.1, vibrance=1.05)
        elif enhancement_mode == "landscape":
            result = PostprocessingMethods.texture_enhancement(result, strength=0.4)
            result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            result = PostprocessingMethods.color_enhancement(result, saturation=1.15, vibrance=1.1)
        elif enhancement_mode == "text":
            result = PostprocessingMethods.enhance_edges(result, strength=1.3)
            result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.6)
        elif enhancement_mode == "art":
            result = PostprocessingMethods.texture_enhancement(result, strength=0.3)
            result = PostprocessingMethods.color_enhancement(result, saturation=1.2, vibrance=1.15)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used=f"smart_enhancement_{enhancement_mode}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_quality_boosting(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        boost_level: str = "medium",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale with quality boosting for maximum quality output.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            boost_level: Boost level ('low', 'medium', 'high', 'ultra')
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Define boost parameters
        boost_params = {
            "low": {"iterations": 1, "strength": 0.2},
            "medium": {"iterations": 2, "strength": 0.3},
            "high": {"iterations": 3, "strength": 0.4},
            "ultra": {"iterations": 5, "strength": 0.5},
        }
        
        params = boost_params.get(boost_level, boost_params["medium"])
        
        # Initial upscale with best method
        result = self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        # Apply quality boosting iterations
        for iteration in range(params["iterations"]):
            result = PostprocessingMethods.enhance_edges(result, strength=1.0 + params["strength"] * (iteration + 1) * 0.1)
            result = PostprocessingMethods.texture_enhancement(result, strength=params["strength"] * 0.2)
            result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=params["strength"] * 0.4)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used=f"quality_boosting_{boost_level}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_hybrid_method(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        primary_method: str = "real_esrgan_like",
        secondary_method: str = "lanczos",
        blend_ratio: float = 0.7,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale using hybrid method combining two upscaling techniques.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            primary_method: Primary upscaling method
            secondary_method: Secondary upscaling method
            blend_ratio: Ratio for blending (0.0-1.0, higher = more primary)
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Upscale with both methods
        result1 = self.upscale(pil_image, scale_factor, primary_method, return_metrics=False)
        result2 = self.upscale(pil_image, scale_factor, secondary_method, return_metrics=False)
        
        # Blend results
        arr1 = np.array(result1, dtype=np.float32)
        arr2 = np.array(result2, dtype=np.float32)
        blended = blend_ratio * arr1 + (1 - blend_ratio) * arr2
        result = Image.fromarray(blended.astype(np.uint8))
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used=f"hybrid_{primary_method}_{secondary_method}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_adaptive_quality_control(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        target_quality: float = 0.85,
        max_iterations: int = 5,
        quality_tolerance: float = 0.05,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale with adaptive quality control loop.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            target_quality: Target quality score
            max_iterations: Maximum iterations
            quality_tolerance: Quality tolerance for stopping
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Start with recommended method
        method = MethodSelector.select_best_method(pil_image, scale_factor)
        result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
        best_result = result
        best_quality = 0.0
        
        # Adaptive quality control loop
        for iteration in range(max_iterations):
            quality = QualityCalculator.calculate_quality_metrics(result)
            current_quality = quality.overall_quality
            
            if current_quality > best_quality:
                best_quality = current_quality
                best_result = result
            
            if abs(current_quality - target_quality) <= quality_tolerance:
                result = best_result
                break
            
            # Apply adaptive enhancements
            if current_quality < target_quality:
                if quality.sharpness < 500:
                    result = PostprocessingMethods.enhance_edges(result, strength=1.1 + iteration * 0.05)
                if quality.contrast < 30:
                    result = PostprocessingMethods.adaptive_contrast_enhancement(result)
                if quality.noise_level > 10:
                    result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.5)
        
        result = best_result
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used=f"{method}_adaptive_qc",
            success=True,
        )
        
        if abs(quality_metrics.overall_quality - target_quality) > quality_tolerance:
            metrics.warnings.append(
                f"Target quality {target_quality:.2f} not fully reached, achieved {quality_metrics.overall_quality:.2f}"
            )
        
        if return_metrics:
            return result, metrics
        return result



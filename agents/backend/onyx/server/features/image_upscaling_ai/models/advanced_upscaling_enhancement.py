"""
Advanced Enhancement Methods
=============================

AI-guided enhancement, quality assurance, and multi-pass processing.
"""

import logging
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image

from .helpers import (
    UpscalingMetrics,
    ImageAnalysisUtils,
    ImageProcessingUtils,
)

logger = logging.getLogger(__name__)


class EnhancementMethods:
    """Advanced enhancement methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
    
    def upscale_with_ai_guided_enhancement(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        enhancement_guidance: str = "auto",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with AI-guided enhancement."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Analyze image for AI guidance
        analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
        
        # Auto-determine guidance
        if enhancement_guidance == "auto":
            edge_density = analysis.get("edge_analysis", {}).get("edge_density", 0.1)
            quality = analysis.get("quality_metrics", {}).get("overall_quality", 0.5)
            
            if edge_density > 0.2:
                enhancement_guidance = "optimize_text"
            elif quality > 0.7:
                enhancement_guidance = "enhance_artistic"
            else:
                enhancement_guidance = "preserve_details"
        
        # Initial upscale
        result = self.base_upscaler.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        # Apply guidance-specific enhancements
        if enhancement_guidance == "preserve_details":
            result = self.base_upscaler.enhance_edges(result, strength=1.2)
            result = ImageProcessingUtils.texture_enhancement(result, strength=0.4)
            result = ImageProcessingUtils.frequency_enhancement(result, strength=0.3)
        elif enhancement_guidance == "enhance_artistic":
            result = ImageProcessingUtils.color_enhancement(result, saturation=1.2, vibrance=1.15)
            result = ImageProcessingUtils.texture_enhancement(result, strength=0.3)
            result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
        elif enhancement_guidance == "optimize_text":
            result = self.base_upscaler.enhance_edges(result, strength=1.4)
            result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
            result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
            result = self.base_upscaler.apply_anti_aliasing(result, strength=0.3)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"ai_guided_{enhancement_guidance}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_quality_assurance(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        min_quality_threshold: float = 0.8,
        max_iterations: int = 5,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with quality assurance."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Start with best method
        result = self.base_upscaler.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        best_result = result
        best_quality = 0.0
        iteration = 0
        
        while iteration < max_iterations:
            quality = self.base_upscaler.calculate_quality_metrics(result)
            current_quality = quality.overall_quality
            
            # Track best result
            if current_quality > best_quality:
                best_quality = current_quality
                best_result = result.copy()
            
            # Check if threshold reached
            if current_quality >= min_quality_threshold:
                result = best_result
                break
            
            # Calculate improvement needed
            quality_gap = min_quality_threshold - current_quality
            enhancement_strength = min(1.0, quality_gap * 2.0)
            
            # Apply targeted enhancements
            if quality.sharpness < 600:
                result = self.base_upscaler.enhance_edges(result, strength=1.0 + enhancement_strength * 0.2)
            
            if quality.contrast < 35:
                result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
            
            if quality.noise_level > 10:
                result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
            
            result = ImageProcessingUtils.frequency_enhancement(result, strength=enhancement_strength * 0.3)
            result = ImageProcessingUtils.texture_enhancement(result, strength=enhancement_strength * 0.2)
            
            iteration += 1
        
        result = best_result
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"quality_assurance_{iteration}",
            success=True,
        )
        
        if quality_metrics.overall_quality < min_quality_threshold:
            metrics.warnings.append(
                f"Quality threshold {min_quality_threshold:.2f} not fully reached, achieved {quality_metrics.overall_quality:.2f}"
            )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_multi_pass_processing(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        passes: int = 3,
        pass_methods: Optional[list] = None,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with multi-pass processing."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        if pass_methods is None:
            pass_methods = ["lanczos", "bicubic", "opencv"]
        
        # Calculate scale per pass
        scale_per_pass = scale_factor ** (1.0 / passes)
        result = pil_image
        
        # Multi-pass upscaling
        for i, method in enumerate(pass_methods[:passes]):
            result = self.base_upscaler.upscale(result, scale_per_pass, method, return_metrics=False)
            
            # Apply pass-specific enhancements
            if i < passes - 1:
                result = self.base_upscaler.enhance_edges(result, strength=1.1)
                result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"multi_pass_{passes}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_advanced_processing(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        use_frequency_analysis: bool = False,
        use_adaptive_contrast: bool = True,
        use_texture_enhancement: bool = True,
        use_color_enhancement: bool = False,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with advanced processing techniques."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Pre-processing
        if use_adaptive_contrast:
            pil_image = ImageProcessingUtils.adaptive_contrast_enhancement(pil_image)
        
        # Upscale
        result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
        
        # Post-processing
        if use_frequency_analysis:
            result = ImageProcessingUtils.frequency_enhancement(result, strength=0.3)
        
        if use_texture_enhancement:
            result = ImageProcessingUtils.texture_enhancement(result, strength=0.2)
        
        if use_color_enhancement:
            result = ImageProcessingUtils.color_enhancement(result, saturation=1.05, vibrance=1.02)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"{method}_advanced",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result



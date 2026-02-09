"""
Advanced Adaptive Processing Methods
====================================

Adaptive and region-based processing methods.
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from PIL import Image

from .helpers import (
    UpscalingMetrics,
    ImageAnalysisUtils,
    ImageProcessingUtils,
)

logger = logging.getLogger(__name__)


class AdaptiveMethods:
    """Adaptive processing methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
    
    def upscale_with_progressive_enhancement(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        enhancement_steps: int = 3,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with progressive enhancement steps."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Initial upscale
        result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
        
        # Progressive enhancement
        for step in range(enhancement_steps):
            quality = self.base_upscaler.calculate_quality_metrics(result)
            
            # Apply enhancements based on quality
            if quality.sharpness < 500:
                result = self.base_upscaler.enhance_edges(result, strength=1.1)
            
            if quality.contrast < 30:
                result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
            
            if quality.noise_level > 10:
                result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
            
            if step < enhancement_steps - 1:
                result = ImageProcessingUtils.texture_enhancement(result, strength=0.1)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"{method}_progressive",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_adaptive_regions(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        min_region_size: int = 256,
        max_region_size: int = 1024,
        quality_threshold: float = 0.7,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with adaptive region sizing."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Analyze image
        analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
        quality = analysis.get("quality_metrics", {}).get("overall_quality", 0.5)
        edge_density = analysis.get("edge_analysis", {}).get("edge_density", 0.1)
        
        # Calculate adaptive region size
        if quality < quality_threshold or edge_density > 0.15:
            region_size = min_region_size
        elif edge_density < 0.05:
            region_size = max_region_size
        else:
            complexity = (quality_threshold - quality) / quality_threshold + edge_density
            region_size = int(min_region_size + (max_region_size - min_region_size) * (1 - complexity))
        
        # Use regional upscaling (simplified - would need full implementation)
        result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"{method}_adaptive_regions",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_adaptive_quality_loop(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        target_quality: float = 0.9,
        max_iterations: int = 10,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with adaptive quality loop."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Initial upscale
        result = self.base_upscaler.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        # Adaptive quality loop
        for iteration in range(max_iterations):
            quality = self.base_upscaler.calculate_quality_metrics(result)
            
            if quality.overall_quality >= target_quality:
                break
            
            # Apply enhancements based on what's needed
            if quality.sharpness < 600:
                result = self.base_upscaler.enhance_edges(result, strength=1.1)
            
            if quality.contrast < 35:
                result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
            
            if quality.noise_level > 10:
                result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
            
            # Frequency enhancement
            result = ImageProcessingUtils.frequency_enhancement(result, strength=0.1)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"adaptive_quality_loop_{iteration+1}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_progressive_quality(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        quality_steps: int = 3,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with progressive quality improvement."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Initial upscale
        result = self.base_upscaler.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        # Progressive quality steps
        for step in range(quality_steps):
            quality = self.base_upscaler.calculate_quality_metrics(result)
            improvement_factor = (quality_steps - step) / quality_steps
            
            # Apply progressive enhancements
            if quality.sharpness < 600:
                result = self.base_upscaler.enhance_edges(result, strength=1.0 + improvement_factor * 0.2)
            
            if quality.contrast < 35:
                result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
            
            result = ImageProcessingUtils.frequency_enhancement(result, strength=0.1 * improvement_factor)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"progressive_quality_{quality_steps}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_region_adaptive_processing(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with region-adaptive processing."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Analyze regions
        analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
        
        # Select method based on analysis
        if analysis.get("is_anime", False):
            method = "waifu2x_like"
        elif analysis.get("is_photo", False):
            method = "real_esrgan_like"
        else:
            method = "esrgan_like"
        
        result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
        
        # Adaptive post-processing
        edge_density = analysis.get("edge_analysis", {}).get("edge_density", 0.1)
        if edge_density > 0.15:
            result = self.base_upscaler.enhance_edges(result, strength=1.2)
        
        quality = analysis.get("quality_metrics", {}).get("overall_quality", 0.5)
        if quality < 0.6:
            result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used="region_adaptive",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_with_adaptive_method_selection(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        candidate_methods: Optional[List[str]] = None,
        selection_criteria: str = "balanced",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale with adaptive method selection."""
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        if candidate_methods is None:
            candidate_methods = ["lanczos", "bicubic", "opencv", "multi_scale", "esrgan_like"]
        
        # Evaluate candidates
        candidates = []
        for method in candidate_methods:
            try:
                method_start = time.time()
                result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
                method_time = time.time() - method_start
                
                quality = self.base_upscaler.calculate_quality_metrics(result)
                
                # Calculate score
                if selection_criteria == "quality":
                    score = quality.overall_quality
                elif selection_criteria == "speed":
                    score = 1.0 / (method_time + 0.001)
                else:  # balanced
                    score = quality.overall_quality * 0.7 + (1.0 / (method_time + 0.001)) * 0.3
                
                candidates.append({
                    "method": method,
                    "result": result,
                    "quality": quality.overall_quality,
                    "time": method_time,
                    "score": score,
                })
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        if not candidates:
            raise RuntimeError("All candidate methods failed")
        
        # Select best
        best_candidate = max(candidates, key=lambda x: x["score"])
        result = best_candidate["result"]
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"adaptive_selection_{best_candidate['method']}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result



"""
Quality Assurance Mixin

Contains quality assurance and validation methods.
"""

import logging
from typing import Union, Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityCalculator,
    ImageQualityValidator,
    PostprocessingMethods,
)

logger = logging.getLogger(__name__)


class QualityAssuranceMixin:
    """
    Mixin providing quality assurance functionality.
    
    This mixin contains:
    - Quality validation
    - Quality improvement
    - Quality comparison
    - Quality reporting
    - Automatic quality enhancement
    """
    
    def validate_upscale_quality(
        self,
        original: Image.Image,
        upscaled: Image.Image,
        min_quality: float = 0.7,
        min_improvement: float = 0.0
    ) -> Dict[str, Any]:
        """
        Validate upscaled image quality.
        
        Args:
            original: Original image
            upscaled: Upscaled image
            min_quality: Minimum acceptable quality
            min_improvement: Minimum quality improvement required
            
        Returns:
            Dictionary with validation results
        """
        original_quality = QualityCalculator.calculate_quality_metrics(original)
        upscaled_quality = QualityCalculator.calculate_quality_metrics(upscaled)
        
        quality_improvement = upscaled_quality.overall_quality - original_quality.overall_quality
        
        is_valid = (
            upscaled_quality.overall_quality >= min_quality and
            quality_improvement >= min_improvement
        )
        
        return {
            "is_valid": is_valid,
            "original_quality": original_quality.overall_quality,
            "upscaled_quality": upscaled_quality.overall_quality,
            "quality_improvement": quality_improvement,
            "min_quality": min_quality,
            "min_improvement": min_improvement,
            "passed": is_valid,
            "warnings": [] if is_valid else [
                f"Quality below minimum: {upscaled_quality.overall_quality:.3f} < {min_quality:.3f}",
                f"Insufficient improvement: {quality_improvement:.3f} < {min_improvement:.3f}"
            ] if quality_improvement < min_improvement else [
                f"Quality below minimum: {upscaled_quality.overall_quality:.3f} < {min_quality:.3f}"
            ]
        }
    
    def upscale_with_quality_assurance(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        min_quality: float = 0.8,
        max_iterations: int = 3,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale with quality assurance - ensures minimum quality.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            min_quality: Minimum quality threshold
            max_iterations: Maximum enhancement iterations
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        import time
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Initial upscale
        result = self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        best_result = result
        best_quality = 0.0
        
        # Quality assurance loop
        for iteration in range(max_iterations):
            quality = QualityCalculator.calculate_quality_metrics(result)
            current_quality = quality.overall_quality
            
            if current_quality > best_quality:
                best_quality = current_quality
                best_result = result
            
            if current_quality >= min_quality:
                result = best_result
                break
            
            # Apply enhancements
            if quality.sharpness < 500:
                result = PostprocessingMethods.enhance_edges(result, strength=1.1 + iteration * 0.1)
            if quality.contrast < 30:
                result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            if quality.noise_level > 10:
                result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.5)
            result = PostprocessingMethods.texture_enhancement(result, strength=0.2)
        
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
            method_used="quality_assurance",
            success=quality_metrics.overall_quality >= min_quality,
        )
        
        if quality_metrics.overall_quality < min_quality:
            metrics.warnings.append(
                f"Target quality {min_quality:.2f} not fully reached, achieved {quality_metrics.overall_quality:.2f}"
            )
        
        if return_metrics:
            return result, metrics
        return result
    
    def compare_quality(
        self,
        images: Dict[str, Image.Image],
        reference: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Compare quality of multiple images.
        
        Args:
            images: Dictionary of {name: image} pairs
            reference: Optional reference image for comparison
            
        Returns:
            Dictionary with quality comparison results
        """
        results = {}
        
        for name, img in images.items():
            quality = QualityCalculator.calculate_quality_metrics(img)
            results[name] = {
                "overall_quality": quality.overall_quality,
                "sharpness": quality.sharpness,
                "contrast": quality.contrast,
                "brightness": quality.brightness,
                "noise_level": quality.noise_level,
                "artifact_count": quality.artifact_count,
            }
        
        # Find best
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]["overall_quality"])
            results["best"] = best_name
            results["comparison"] = {
                "best_quality": results[best_name]["overall_quality"],
                "worst_quality": min(r["overall_quality"] for r in results.values()),
                "quality_range": max(r["overall_quality"] for r in results.values()) - min(r["overall_quality"] for r in results.values())
            }
        
        return results
    
    def get_quality_report(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method: Upscaling method or 'auto'
            
        Returns:
            Dictionary with quality report
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_quality = QualityCalculator.calculate_quality_metrics(pil_image)
        
        if method == "auto":
            method = MethodSelector.select_best_method(pil_image, scale_factor)
        
        result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
        upscaled_quality = QualityCalculator.calculate_quality_metrics(result)
        
        quality_improvement = upscaled_quality.overall_quality - original_quality.overall_quality
        
        return {
            "original": {
                "size": pil_image.size,
                "quality": original_quality.overall_quality,
                "sharpness": original_quality.sharpness,
                "contrast": original_quality.contrast,
            },
            "upscaled": {
                "size": result.size,
                "quality": upscaled_quality.overall_quality,
                "sharpness": upscaled_quality.sharpness,
                "contrast": upscaled_quality.contrast,
            },
            "improvement": {
                "quality": quality_improvement,
                "sharpness": upscaled_quality.sharpness - original_quality.sharpness,
                "contrast": upscaled_quality.contrast - original_quality.contrast,
            },
            "method_used": method,
            "scale_factor": scale_factor,
            "recommendations": self._get_quality_recommendations(upscaled_quality)
        }
    
    def _get_quality_recommendations(self, quality) -> List[str]:
        """Get quality improvement recommendations."""
        recommendations = []
        
        if quality.sharpness < 500:
            recommendations.append("Apply edge enhancement to improve sharpness")
        if quality.contrast < 30:
            recommendations.append("Apply contrast enhancement")
        if quality.noise_level > 10:
            recommendations.append("Apply noise reduction")
        if quality.artifact_count > 0.1:
            recommendations.append("Apply artifact reduction")
        
        if not recommendations:
            recommendations.append("Image quality is excellent")
        
        return recommendations



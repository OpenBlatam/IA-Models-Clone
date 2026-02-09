"""
Validation Mixin

Contains advanced validation and verification functionality.
"""

import logging
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image

from ..helpers import (
    ImageQualityValidator,
    QualityCalculator,
)

logger = logging.getLogger(__name__)


class ValidationMixin:
    """
    Mixin providing advanced validation functionality.
    
    This mixin contains:
    - Image validation
    - Quality validation
    - Method validation
    - Result validation
    - Batch validation
    """
    
    def validate_image(
        self,
        image: Union[Image.Image, str, Path],
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate image with comprehensive checks.
        
        Args:
            image: Input image or path
            strict: Use strict validation
            
        Returns:
            Dictionary with validation results
        """
        if isinstance(image, (str, Path)):
            try:
                pil_image = Image.open(image).convert("RGB")
                file_path = str(image)
            except Exception as e:
                return {
                    "is_valid": False,
                    "errors": [f"Failed to open image: {str(e)}"],
                    "warnings": [],
                    "file_path": str(image)
                }
        else:
            pil_image = image.convert("RGB")
            file_path = None
        
        # Basic validation
        validation = ImageQualityValidator.validate_image(pil_image)
        
        # Additional checks
        errors = list(validation.errors) if validation.errors else []
        warnings = list(validation.warnings) if validation.warnings else []
        
        # Size checks
        width, height = pil_image.size
        if width < 10 or height < 10:
            errors.append("Image too small (minimum 10x10 pixels)")
        elif width > 10000 or height > 10000:
            warnings.append("Image very large, may cause performance issues")
        
        # Format checks
        if pil_image.mode not in ["RGB", "RGBA", "L"]:
            warnings.append(f"Image mode {pil_image.mode} may not be optimal")
        
        # Quality checks
        quality = QualityCalculator.calculate_quality_metrics(pil_image)
        if quality.overall_quality < 0.3:
            warnings.append("Image quality is very low")
        
        is_valid = len(errors) == 0 if strict else validation.is_valid
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "file_path": file_path,
            "size": pil_image.size,
            "mode": pil_image.mode,
            "quality_score": quality.overall_quality,
        }
    
    def validate_upscale_result(
        self,
        original: Image.Image,
        upscaled: Image.Image,
        scale_factor: float,
        min_improvement: float = 0.0
    ) -> Dict[str, Any]:
        """
        Validate upscaling result.
        
        Args:
            original: Original image
            upscaled: Upscaled image
            scale_factor: Scale factor used
            min_improvement: Minimum quality improvement required
            
        Returns:
            Dictionary with validation results
        """
        original_quality = QualityCalculator.calculate_quality_metrics(original)
        upscaled_quality = QualityCalculator.calculate_quality_metrics(upscaled)
        
        # Check size
        expected_width = int(original.size[0] * scale_factor)
        expected_height = int(original.size[1] * scale_factor)
        actual_width, actual_height = upscaled.size
        
        size_correct = (
            abs(actual_width - expected_width) <= 1 and
            abs(actual_height - expected_height) <= 1
        )
        
        # Check quality improvement
        quality_improvement = upscaled_quality.overall_quality - original_quality.overall_quality
        quality_acceptable = quality_improvement >= min_improvement
        
        # Check for degradation
        quality_degraded = quality_improvement < -0.1
        
        errors = []
        warnings = []
        
        if not size_correct:
            errors.append(
                f"Size mismatch: expected {expected_width}x{expected_height}, "
                f"got {actual_width}x{actual_height}"
            )
        
        if quality_degraded:
            errors.append(
                f"Quality degraded by {abs(quality_improvement):.3f}"
            )
        elif not quality_acceptable:
            warnings.append(
                f"Quality improvement {quality_improvement:.3f} below minimum {min_improvement:.3f}"
            )
        
        if upscaled_quality.artifact_count > 0.2:
            warnings.append("High artifact count detected")
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "size_correct": size_correct,
            "quality_improvement": quality_improvement,
            "quality_acceptable": quality_acceptable,
            "original_quality": original_quality.overall_quality,
            "upscaled_quality": upscaled_quality.overall_quality,
        }
    
    def validate_method(
        self,
        method: str,
        scale_factor: float
    ) -> Dict[str, Any]:
        """
        Validate if method is suitable for scale factor.
        
        Args:
            method: Upscaling method
            scale_factor: Scale factor
            
        Returns:
            Dictionary with validation results
        """
        valid_methods = [
            "lanczos", "bicubic", "opencv", "multi_scale",
            "adaptive", "esrgan_like", "waifu2x_like", "real_esrgan_like"
        ]
        
        errors = []
        warnings = []
        
        if method not in valid_methods:
            errors.append(f"Unknown method: {method}")
        
        if scale_factor < 1.0:
            errors.append("Scale factor must be >= 1.0")
        elif scale_factor > 10.0:
            warnings.append("Very high scale factor, may cause performance issues")
        
        if method == "lanczos" and scale_factor > 4.0:
            warnings.append("Lanczos may not be optimal for scale factors > 4x")
        
        if method in ["esrgan_like", "waifu2x_like", "real_esrgan_like"] and scale_factor < 2.0:
            warnings.append("ML methods are optimized for 2x+ scale factors")
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "method": method,
            "scale_factor": scale_factor,
            "recommended": method in ["real_esrgan_like", "esrgan_like"] if scale_factor >= 2.0 else method == "lanczos"
        }
    
    def batch_validate_images(
        self,
        images: List[Union[Image.Image, str, Path]],
        strict: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple images.
        
        Args:
            images: List of images or paths
            strict: Use strict validation
            
        Returns:
            List of validation results
        """
        return [self.validate_image(img, strict=strict) for img in images]
    
    def get_validation_summary(
        self,
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Args:
            validation_results: List of validation result dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        total = len(validation_results)
        valid = sum(1 for r in validation_results if r.get("is_valid", False))
        invalid = total - valid
        
        total_errors = sum(len(r.get("errors", [])) for r in validation_results)
        total_warnings = sum(len(r.get("warnings", [])) for r in validation_results)
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "validity_rate": valid / total if total > 0 else 0.0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "avg_errors_per_image": total_errors / total if total > 0 else 0.0,
            "avg_warnings_per_image": total_warnings / total if total > 0 else 0.0,
        }



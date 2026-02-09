"""
Quality Validator
================

Comprehensive quality validation for upscaled images.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class QualityReport:
    """Quality validation report."""
    passed: bool
    overall_score: float  # 0.0-1.0
    issues: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]


class QualityValidator:
    """
    Comprehensive quality validation for upscaled images.
    
    Features:
    - Artifact detection
    - Sharpness validation
    - Color accuracy
    - Edge quality
    - Noise assessment
    - Comparison with original
    """
    
    def __init__(
        self,
        min_score: float = 0.6,
        strict_mode: bool = False
    ):
        """
        Initialize validator.
        
        Args:
            min_score: Minimum acceptable quality score
            strict_mode: Use stricter validation criteria
        """
        self.min_score = min_score
        self.strict_mode = strict_mode
    
    def validate(
        self,
        upscaled: Image.Image,
        original: Optional[Image.Image] = None,
        scale_factor: Optional[float] = None
    ) -> QualityReport:
        """
        Validate upscaled image quality.
        
        Args:
            upscaled: Upscaled image
            original: Original image (for comparison)
            scale_factor: Scale factor used
            
        Returns:
            QualityReport with validation results
        """
        issues = []
        recommendations = []
        metrics = {}
        
        # Basic checks
        if upscaled.size[0] < 64 or upscaled.size[1] < 64:
            issues.append("Upscaled image is too small")
        
        # Calculate metrics
        metrics["sharpness"] = self._measure_sharpness(upscaled)
        metrics["artifacts"] = self._detect_artifacts(upscaled)
        metrics["noise"] = self._measure_noise(upscaled)
        metrics["contrast"] = self._measure_contrast(upscaled)
        
        # Compare with original if available
        if original:
            metrics["ssim"] = self._calculate_ssim(upscaled, original, scale_factor)
            metrics["psnr"] = self._calculate_psnr(upscaled, original, scale_factor)
            metrics["color_accuracy"] = self._measure_color_accuracy(upscaled, original, scale_factor)
        
        # Check thresholds
        if metrics["sharpness"] < 0.5:
            issues.append(f"Low sharpness: {metrics['sharpness']:.2f}")
            recommendations.append("Apply edge enhancement")
        
        if metrics["artifacts"] > 0.4:
            issues.append(f"High artifact level: {metrics['artifacts']:.2f}")
            recommendations.append("Apply artifact reduction")
        
        if metrics["noise"] > 0.5:
            issues.append(f"High noise level: {metrics['noise']:.2f}")
            recommendations.append("Apply denoising")
        
        if original:
            if metrics.get("ssim", 0) < 0.7:
                issues.append(f"Low SSIM: {metrics['ssim']:.2f}")
                recommendations.append("Improve upscaling method")
            
            if metrics.get("psnr", 0) < 20:
                issues.append(f"Low PSNR: {metrics['psnr']:.2f} dB")
                recommendations.append("Improve upscaling quality")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics, original is not None)
        
        # Determine if passed
        passed = overall_score >= self.min_score
        if self.strict_mode and issues:
            passed = False
        
        return QualityReport(
            passed=passed,
            overall_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _measure_sharpness(self, image: Image.Image) -> float:
        """Measure image sharpness (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.7  # Default
        
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var() / 3000.0
        
        return min(1.0, max(0.0, sharpness))
    
    def _detect_artifacts(self, image: Image.Image) -> float:
        """Detect artifacts (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.2  # Default
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect block artifacts
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = np.abs(gray.astype(float) - blurred.astype(float))
        artifact_score = np.std(diff) / 255.0
        
        return min(1.0, artifact_score)
    
    def _measure_noise(self, image: Image.Image) -> float:
        """Measure noise level (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.3  # Default
        
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.std(gray.astype(float) - blurred.astype(float)) / 255.0
        
        return min(1.0, noise)
    
    def _measure_contrast(self, image: Image.Image) -> float:
        """Measure contrast (0.0-1.0)."""
        img_array = np.array(image)
        std = np.std(img_array, axis=(0, 1))
        contrast = np.mean(std) / 255.0
        
        return min(1.0, contrast)
    
    def _calculate_ssim(
        self,
        upscaled: Image.Image,
        original: Image.Image,
        scale_factor: Optional[float]
    ) -> float:
        """Calculate SSIM (Structural Similarity Index)."""
        # Resize original to match upscaled
        if scale_factor:
            orig_size = (
                int(original.size[0] * scale_factor),
                int(original.size[1] * scale_factor)
            )
        else:
            orig_size = upscaled.size
        
        orig_resized = original.resize(orig_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale for SSIM
        upscaled_gray = np.array(upscaled.convert("L"), dtype=np.float32)
        orig_gray = np.array(orig_resized.convert("L"), dtype=np.float32)
        
        # Simple SSIM calculation
        mu1 = np.mean(upscaled_gray)
        mu2 = np.mean(orig_gray)
        
        sigma1_sq = np.var(upscaled_gray)
        sigma2_sq = np.var(orig_gray)
        sigma12 = np.mean((upscaled_gray - mu1) * (orig_gray - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = (
            (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) /
            ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        )
        
        return max(0.0, min(1.0, ssim))
    
    def _calculate_psnr(
        self,
        upscaled: Image.Image,
        original: Image.Image,
        scale_factor: Optional[float]
    ) -> float:
        """Calculate PSNR (Peak Signal-to-Noise Ratio) in dB."""
        # Resize original to match upscaled
        if scale_factor:
            orig_size = (
                int(original.size[0] * scale_factor),
                int(original.size[1] * scale_factor)
            )
        else:
            orig_size = upscaled.size
        
        orig_resized = original.resize(orig_size, Image.Resampling.LANCZOS)
        
        upscaled_array = np.array(upscaled.convert("RGB"), dtype=np.float32)
        orig_array = np.array(orig_resized.convert("RGB"), dtype=np.float32)
        
        mse = np.mean((upscaled_array - orig_array) ** 2)
        
        if mse == 0:
            return 100.0  # Perfect match
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def _measure_color_accuracy(
        self,
        upscaled: Image.Image,
        original: Image.Image,
        scale_factor: Optional[float]
    ) -> float:
        """Measure color accuracy (0.0-1.0)."""
        # Resize original to match upscaled
        if scale_factor:
            orig_size = (
                int(original.size[0] * scale_factor),
                int(original.size[1] * scale_factor)
            )
        else:
            orig_size = upscaled.size
        
        orig_resized = original.resize(orig_size, Image.Resampling.LANCZOS)
        
        upscaled_array = np.array(upscaled.convert("RGB"), dtype=np.float32)
        orig_array = np.array(orig_resized.convert("RGB"), dtype=np.float32)
        
        # Calculate color difference
        diff = np.abs(upscaled_array - orig_array)
        mean_diff = np.mean(diff) / 255.0
        
        # Accuracy is inverse of difference
        accuracy = 1.0 - mean_diff
        
        return max(0.0, min(1.0, accuracy))
    
    def _calculate_overall_score(
        self,
        metrics: Dict[str, float],
        has_original: bool
    ) -> float:
        """Calculate overall quality score."""
        # Weighted average
        weights = {
            "sharpness": 0.25,
            "artifacts": -0.20,  # Negative (lower is better)
            "noise": -0.15,  # Negative
            "contrast": 0.10,
        }
        
        if has_original:
            weights["ssim"] = 0.20
            weights["psnr"] = 0.15
            weights["color_accuracy"] = 0.15
        
        score = 0.0
        for key, weight in weights.items():
            if key in metrics:
                value = metrics[key]
                if weight < 0:
                    # For negative weights, invert the value
                    value = 1.0 - value
                    weight = abs(weight)
                score += value * weight
        
        # Normalize
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            score = score / total_weight
        
        return max(0.0, min(1.0, score))



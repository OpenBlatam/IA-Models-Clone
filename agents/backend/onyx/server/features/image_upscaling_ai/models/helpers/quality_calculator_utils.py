"""
Quality Calculator Utilities
============================

Utilities for calculating quality metrics for images.
"""

import logging
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .metrics_utils import QualityMetrics

logger = logging.getLogger(__name__)


class QualityCalculator:
    """Calculator for image quality metrics."""
    
    @staticmethod
    def calculate_quality_metrics(image: Image.Image) -> QualityMetrics:
        """
        Calculate quality metrics for an image.
        
        Args:
            image: Input image
            
        Returns:
            QualityMetrics object
        """
        img_array = np.array(image)
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Calculate sharpness (variance of Laplacian)
        try:
            if CV2_AVAILABLE:
                laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F)
                sharpness = float(np.var(laplacian))
            else:
                # Fallback: gradient-based sharpness
                grad_x = np.gradient(gray.astype(float), axis=1)
                grad_y = np.gradient(gray.astype(float), axis=0)
                sharpness = float(np.var(grad_x) + np.var(grad_y))
        except Exception:
            sharpness = 0.0
        
        # Calculate contrast (standard deviation)
        contrast = float(np.std(gray))
        
        # Calculate brightness (mean)
        brightness = float(np.mean(gray))
        
        # Estimate noise level (variance in smooth regions)
        try:
            if CV2_AVAILABLE:
                # Use median filter to estimate noise
                median = cv2.medianBlur(gray.astype(np.uint8), 5)
                noise = float(np.std(gray.astype(float) - median.astype(float)))
            else:
                noise = contrast * 0.1  # Rough estimate
        except Exception:
            noise = 0.0
        
        # Estimate artifacts (edge artifacts in upscaled images)
        artifact_count = 0.0
        try:
            if CV2_AVAILABLE:
                # Detect edges and check for artifacts
                edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
                artifact_count = float(np.sum(edges > 0) / edges.size)
        except Exception:
            artifact_count = 0.0
        
        # Overall quality score (normalized 0-1)
        sharpness_norm = min(1.0, sharpness / 1000.0)
        contrast_norm = min(1.0, contrast / 50.0)
        brightness_norm = min(1.0, abs(brightness - 128) / 128.0)
        noise_norm = max(0.0, 1.0 - (noise / 20.0))
        artifact_norm = max(0.0, 1.0 - artifact_count)
        
        overall_quality = (
            sharpness_norm * 0.3 +
            contrast_norm * 0.2 +
            brightness_norm * 0.1 +
            noise_norm * 0.2 +
            artifact_norm * 0.2
        )
        
        return QualityMetrics(
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            noise_level=noise,
            artifact_count=artifact_count,
            overall_quality=overall_quality
        )



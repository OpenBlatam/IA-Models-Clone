"""
Quality Calculator
==================

Calculate quality metrics for images.
"""

import logging
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .metrics import QualityMetrics

logger = logging.getLogger(__name__)


class QualityCalculator:
    """Calculate quality metrics for images."""
    
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
        gray = QualityCalculator._to_grayscale(img_array)
        
        # Calculate metrics
        sharpness = QualityCalculator._calculate_sharpness(gray)
        contrast = QualityCalculator._calculate_contrast(gray)
        brightness = QualityCalculator._calculate_brightness(gray)
        noise = QualityCalculator._calculate_noise(gray)
        artifact_count = QualityCalculator._calculate_artifacts(gray)
        
        # Calculate overall quality score
        overall_quality = QualityCalculator._calculate_overall_quality(
            sharpness, contrast, brightness, noise, artifact_count
        )
        
        return QualityMetrics(
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            noise_level=noise,
            artifact_count=artifact_count,
            overall_quality=overall_quality
        )
    
    @staticmethod
    def _to_grayscale(img_array: np.ndarray) -> np.ndarray:
        """Convert image array to grayscale."""
        if len(img_array.shape) == 3:
            return np.mean(img_array, axis=2)
        return img_array
    
    @staticmethod
    def _calculate_sharpness(gray: np.ndarray) -> float:
        """Calculate image sharpness."""
        try:
            if CV2_AVAILABLE:
                laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F)
                return float(np.var(laplacian))
            else:
                # Fallback: gradient-based sharpness
                grad_x = np.gradient(gray.astype(float), axis=1)
                grad_y = np.gradient(gray.astype(float), axis=0)
                return float(np.var(grad_x) + np.var(grad_y))
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_contrast(gray: np.ndarray) -> float:
        """Calculate image contrast."""
        return float(np.std(gray))
    
    @staticmethod
    def _calculate_brightness(gray: np.ndarray) -> float:
        """Calculate image brightness."""
        return float(np.mean(gray))
    
    @staticmethod
    def _calculate_noise(gray: np.ndarray) -> float:
        """Estimate noise level."""
        try:
            if CV2_AVAILABLE:
                # Use median filter to estimate noise
                median = cv2.medianBlur(gray.astype(np.uint8), 5)
                return float(np.std(gray.astype(float) - median.astype(float)))
            else:
                # Rough estimate based on contrast
                contrast = QualityCalculator._calculate_contrast(gray)
                return contrast * 0.1
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_artifacts(gray: np.ndarray) -> float:
        """Estimate artifact count."""
        try:
            if CV2_AVAILABLE:
                # Detect edges and check for artifacts
                edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
                return float(np.sum(edges > 0) / edges.size)
        except Exception:
            pass
        return 0.0
    
    @staticmethod
    def _calculate_overall_quality(
        sharpness: float,
        contrast: float,
        brightness: float,
        noise: float,
        artifact_count: float
    ) -> float:
        """Calculate overall quality score (normalized 0-1)."""
        sharpness_norm = min(1.0, sharpness / 1000.0)
        contrast_norm = min(1.0, contrast / 50.0)
        brightness_norm = min(1.0, abs(brightness - 128) / 128.0)
        noise_norm = max(0.0, 1.0 - (noise / 20.0))
        artifact_norm = max(0.0, 1.0 - artifact_count)
        
        return (
            sharpness_norm * 0.3 +
            contrast_norm * 0.2 +
            brightness_norm * 0.1 +
            noise_norm * 0.2 +
            artifact_norm * 0.2
        )



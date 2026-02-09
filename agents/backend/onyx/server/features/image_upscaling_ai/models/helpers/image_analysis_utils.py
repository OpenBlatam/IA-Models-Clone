"""
Image Analysis Utilities
========================

Utilities for analyzing image characteristics.
"""

import logging
import numpy as np
from typing import Dict, Any, Union
from pathlib import Path
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .quality_calculator_utils import QualityCalculator

logger = logging.getLogger(__name__)


class ImageAnalysisUtils:
    """Utilities for image analysis."""
    
    @staticmethod
    def analyze_image_characteristics(
        image: Union[Image.Image, str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze image characteristics for optimal processing.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image characteristics
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        img_array = np.array(pil_image)
        
        # Basic metrics
        quality = QualityCalculator.calculate_quality_metrics(pil_image)
        
        # Color analysis
        if len(img_array.shape) == 3:
            mean_color = np.mean(img_array, axis=(0, 1))
            std_color = np.std(img_array, axis=(0, 1))
            color_variance = np.var(img_array, axis=(0, 1))
        else:
            mean_color = np.mean(img_array)
            std_color = np.std(img_array)
            color_variance = np.var(img_array)
        
        # Histogram analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array.astype(np.uint8)
        
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        hist_normalized = hist / hist.sum()
        
        # Calculate histogram metrics
        brightness_peak = np.argmax(hist_normalized)
        contrast_range = np.percentile(gray, 95) - np.percentile(gray, 5)
        
        # Edge density
        edge_density = 0.0
        if CV2_AVAILABLE:
            try:
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
            except Exception:
                edge_density = 0.0
        
        return {
            "quality_metrics": {
                "overall_quality": quality.overall_quality,
                "sharpness": quality.sharpness,
                "contrast": quality.contrast,
                "brightness": quality.brightness,
                "noise_level": quality.noise_level,
                "artifact_count": quality.artifact_count,
            },
            "color_analysis": {
                "mean_color": mean_color.tolist() if isinstance(mean_color, np.ndarray) else float(mean_color),
                "std_color": std_color.tolist() if isinstance(std_color, np.ndarray) else float(std_color),
                "color_variance": color_variance.tolist() if isinstance(color_variance, np.ndarray) else float(color_variance),
            },
            "histogram_analysis": {
                "brightness_peak": int(brightness_peak),
                "contrast_range": float(contrast_range),
                "histogram": hist_normalized.tolist(),
            },
            "edge_analysis": {
                "edge_density": float(edge_density),
            },
            "recommendations": {
                "needs_contrast_enhancement": quality.contrast < 30,
                "needs_sharpening": quality.sharpness < 500,
                "needs_noise_reduction": quality.noise_level > 15,
                "needs_color_enhancement": np.mean(std_color) < 20 if isinstance(std_color, np.ndarray) else std_color < 20,
            }
        }



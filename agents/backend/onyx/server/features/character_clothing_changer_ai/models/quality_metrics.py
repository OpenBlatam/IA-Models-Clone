"""
Quality Metrics for Clothing Changes
====================================

Metrics to evaluate the quality of clothing change results.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class QualityMetrics:
    """Calculate quality metrics for clothing change results."""
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        pass
    
    def calculate_metrics(
        self,
        original_image: Image.Image,
        changed_image: Image.Image,
        mask: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for clothing change.
        
        Args:
            original_image: Original image
            changed_image: Changed image
            mask: Optional mask of changed region
            
        Returns:
            Dict with quality metrics
        """
        metrics = {}
        
        # Convert to numpy arrays
        orig_array = np.array(original_image.convert("RGB"))
        changed_array = np.array(changed_image.convert("RGB"))
        
        # Ensure same size
        if orig_array.shape != changed_array.shape:
            changed_image = changed_image.resize(original_image.size, Image.Resampling.LANCZOS)
            changed_array = np.array(changed_image.convert("RGB"))
        
        # Calculate metrics
        metrics["structural_similarity"] = self._calculate_ssim(orig_array, changed_array)
        metrics["color_consistency"] = self._calculate_color_consistency(orig_array, changed_array, mask)
        metrics["sharpness"] = self._calculate_sharpness(changed_array)
        metrics["brightness_consistency"] = self._calculate_brightness_consistency(orig_array, changed_array, mask)
        
        # Overall quality score
        metrics["overall_quality"] = (
            metrics["structural_similarity"] * 0.3 +
            metrics["color_consistency"] * 0.3 +
            metrics["sharpness"] * 0.2 +
            metrics["brightness_consistency"] * 0.2
        )
        
        return metrics
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (SSIM)."""
        try:
            # Convert to grayscale
            gray1 = np.mean(img1, axis=2)
            gray2 = np.mean(img2, axis=2)
            
            # Normalize
            gray1 = gray1.astype(np.float64) / 255.0
            gray2 = gray2.astype(np.float64) / 255.0
            
            # Constants
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            # Calculate means
            mu1 = np.mean(gray1)
            mu2 = np.mean(gray2)
            
            # Calculate variances and covariance
            sigma1_sq = np.var(gray1)
            sigma2_sq = np.var(gray2)
            sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
            
            # SSIM formula
            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim = numerator / denominator if denominator != 0 else 0.0
            
            return float(np.clip(ssim, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error calculating SSIM: {e}")
            return 0.5
    
    def _calculate_color_consistency(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: Optional[Image.Image] = None,
    ) -> float:
        """Calculate color consistency in unchanged regions."""
        try:
            if mask is None:
                # Use upper region (typically unchanged)
                h = img1.shape[0]
                mask_array = np.zeros((h, img1.shape[1]), dtype=bool)
                mask_array[:int(h * 0.4), :] = True
            else:
                mask_array = np.array(mask.convert("L")) > 128
                mask_array = ~mask_array  # Invert to get unchanged regions
            
            # Extract unchanged regions
            unchanged1 = img1[mask_array]
            unchanged2 = img2[mask_array]
            
            if len(unchanged1) == 0:
                return 0.5
            
            # Calculate mean color difference
            color_diff = np.mean(np.abs(unchanged1.astype(float) - unchanged2.astype(float)))
            
            # Normalize to 0-1 (lower is better, so invert)
            consistency = 1.0 - (color_diff / 255.0)
            
            return float(np.clip(consistency, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error calculating color consistency: {e}")
            return 0.5
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        if not CV2_AVAILABLE:
            # Fallback: simple gradient-based sharpness
            try:
                gray = np.mean(image, axis=2).astype(np.float64)
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                variance = np.var(grad_x) + np.var(grad_y)
                sharpness = min(variance / 1000.0, 1.0)
                return float(np.clip(sharpness, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Error calculating sharpness: {e}")
                return 0.5
        
        try:
            # Convert to grayscale
            gray = np.mean(image, axis=2).astype(np.uint8)
            
            # Calculate Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = np.var(laplacian)
            
            # Normalize (typical sharp images have variance > 100)
            sharpness = min(variance / 500.0, 1.0)
            
            return float(np.clip(sharpness, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error calculating sharpness: {e}")
            return 0.5
    
    def _calculate_brightness_consistency(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: Optional[Image.Image] = None,
    ) -> float:
        """Calculate brightness consistency in unchanged regions."""
        try:
            if mask is None:
                h = img1.shape[0]
                mask_array = np.zeros((h, img1.shape[1]), dtype=bool)
                mask_array[:int(h * 0.4), :] = True
            else:
                mask_array = np.array(mask.convert("L")) > 128
                mask_array = ~mask_array
            
            # Calculate brightness
            brightness1 = np.mean(img1[mask_array])
            brightness2 = np.mean(img2[mask_array])
            
            # Calculate difference
            diff = abs(brightness1 - brightness2) / 255.0
            
            # Consistency (lower diff = higher consistency)
            consistency = 1.0 - diff
            
            return float(np.clip(consistency, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error calculating brightness consistency: {e}")
            return 0.5


# Import cv2 for sharpness calculation
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available, sharpness calculation will be limited")


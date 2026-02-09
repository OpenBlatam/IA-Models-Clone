"""
Quality Metrics for Image Upscaling
===================================

Advanced quality metrics for evaluating upscaled images.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class QualityMetrics:
    """
    Calculate quality metrics for upscaled images.
    
    Metrics:
    - SSIM (Structural Similarity Index)
    - PSNR (Peak Signal-to-Noise Ratio)
    - Sharpness
    - Gradient preservation
    - Artifact detection
    """
    
    @staticmethod
    def calculate_ssim(
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            SSIM score (0.0 to 1.0, higher is better)
        """
        if not CV2_AVAILABLE:
            # Fallback to simple correlation
            return 0.5
        
        try:
            # Ensure same size
            if img1.shape != img2.shape:
                h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                img1 = img1[:h, :w]
                img2 = img2[:h, :w]
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Calculate SSIM
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            
            mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            )
            
            return float(np.mean(ssim_map))
            
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_psnr(
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            PSNR in dB (higher is better, typically 20-50 dB)
        """
        try:
            # Ensure same size
            if img1.shape != img2.shape:
                h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                img1 = img1[:h, :w]
                img2 = img2[:h, :w]
            
            # Calculate MSE
            mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
            
            if mse == 0:
                return 100.0  # Perfect match
            
            # Calculate PSNR
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return float(psnr)
            
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_sharpness(
        image: np.ndarray
    ) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Image array
            
        Returns:
            Sharpness score (higher is sharper)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Variance of Laplacian is a measure of sharpness
            sharpness = float(laplacian.var())
            
            return sharpness
            
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_gradient_preservation(
        original: np.ndarray,
        upscaled: np.ndarray
    ) -> float:
        """
        Calculate how well gradients are preserved.
        
        Args:
            original: Original image array
            upscaled: Upscaled image array
            
        Returns:
            Gradient preservation score (0.0 to 1.0)
        """
        try:
            # Resize original to match upscaled
            if original.shape != upscaled.shape:
                h, w = upscaled.shape[:2]
                if len(original.shape) == 3:
                    original_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    original_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                original_resized = original
            
            # Convert to grayscale if needed
            if len(original_resized.shape) == 3:
                orig_gray = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
                upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = original_resized
                upscaled_gray = upscaled
            
            # Calculate gradients
            grad_orig_x = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_orig_y = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_orig_mag = np.sqrt(grad_orig_x ** 2 + grad_orig_y ** 2)
            
            grad_upscaled_x = cv2.Sobel(upscaled_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_upscaled_y = cv2.Sobel(upscaled_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_upscaled_mag = np.sqrt(grad_upscaled_x ** 2 + grad_upscaled_y ** 2)
            
            # Calculate correlation
            orig_mean = np.mean(grad_orig_mag)
            upscaled_mean = np.mean(grad_upscaled_mag)
            
            if orig_mean == 0:
                return 0.0
            
            # Normalized correlation
            preservation = upscaled_mean / (orig_mean + 1e-6)
            preservation = min(1.0, max(0.0, preservation))
            
            return float(preservation)
            
        except Exception as e:
            logger.warning(f"Gradient preservation calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def detect_artifacts(
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect common upscaling artifacts.
        
        Args:
            image: Image array
            
        Returns:
            Dictionary with artifact detection results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect ringing artifacts (high frequency patterns)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # High frequency energy
            h, w = gray.shape
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (min(h, w) // 4) ** 2
            high_freq_energy = np.mean(magnitude[mask])
            
            # Detect block artifacts (checkerboard patterns)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            block_score = np.std(laplacian)
            
            # Detect blur (low variance in gradients)
            gradients = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            blur_score = 1.0 / (1.0 + np.var(gradients))
            
            return {
                "ringing_artifacts": float(high_freq_energy),
                "block_artifacts": float(block_score),
                "blur_score": float(blur_score),
                "has_artifacts": high_freq_energy > 1000 or block_score > 50
            }
            
        except Exception as e:
            logger.warning(f"Artifact detection failed: {e}")
            return {
                "ringing_artifacts": 0.0,
                "block_artifacts": 0.0,
                "blur_score": 0.0,
                "has_artifacts": False
            }
    
    @staticmethod
    def calculate_all_metrics(
        original: Image.Image,
        upscaled: Image.Image
    ) -> Dict[str, Any]:
        """
        Calculate all quality metrics.
        
        Args:
            original: Original image
            upscaled: Upscaled image
            
        Returns:
            Dictionary with all metrics
        """
        try:
            # Convert to numpy arrays
            orig_array = np.array(original)
            upscaled_array = np.array(upscaled)
            
            # Resize original to match upscaled for comparison
            orig_resized = original.resize(upscaled.size, Image.Resampling.LANCZOS)
            orig_resized_array = np.array(orig_resized)
            
            # Calculate metrics
            ssim = QualityMetrics.calculate_ssim(orig_resized_array, upscaled_array)
            psnr = QualityMetrics.calculate_psnr(orig_resized_array, upscaled_array)
            sharpness = QualityMetrics.calculate_sharpness(upscaled_array)
            gradient_preservation = QualityMetrics.calculate_gradient_preservation(
                orig_resized_array,
                upscaled_array
            )
            artifacts = QualityMetrics.detect_artifacts(upscaled_array)
            
            # Overall quality score (weighted average)
            overall_score = (
                0.3 * ssim +
                0.2 * min(psnr / 50.0, 1.0) +  # Normalize PSNR
                0.2 * min(sharpness / 1000.0, 1.0) +  # Normalize sharpness
                0.2 * gradient_preservation +
                0.1 * (1.0 - artifacts["blur_score"])
            )
            
            return {
                "ssim": ssim,
                "psnr": psnr,
                "sharpness": sharpness,
                "gradient_preservation": gradient_preservation,
                "artifacts": artifacts,
                "overall_quality": float(overall_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}", exc_info=True)
            return {
                "ssim": 0.0,
                "psnr": 0.0,
                "sharpness": 0.0,
                "gradient_preservation": 0.0,
                "artifacts": {},
                "overall_quality": 0.0
            }


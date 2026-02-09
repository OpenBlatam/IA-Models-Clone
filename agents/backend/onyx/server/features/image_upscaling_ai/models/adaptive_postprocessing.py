"""
Adaptive Postprocessing
=======================

Advanced postprocessing that adapts to upscaled image characteristics.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class AdaptivePostprocessor:
    """
    Adaptive postprocessing for upscaled images.
    
    Features:
    - Artifact detection and removal
    - Edge enhancement
    - Ringing reduction
    - Color consistency
    - Quality-based adjustments
    """
    
    def __init__(self):
        """Initialize postprocessor."""
        pass
    
    def postprocess(
        self,
        upscaled: Image.Image,
        original: Optional[Image.Image] = None,
        analysis: Optional[Dict[str, Any]] = None,
        mode: str = "auto"
    ) -> Image.Image:
        """
        Postprocess upscaled image.
        
        Args:
            upscaled: Upscaled image
            original: Original image (for comparison)
            analysis: Quality analysis (auto-analyze if None)
            mode: Postprocessing mode ('auto', 'aggressive', 'conservative', 'none')
            
        Returns:
            Postprocessed image
        """
        if mode == "none":
            return upscaled
        
        processed = upscaled
        
        # Analyze if needed
        if analysis is None:
            analysis = self._analyze_upscaled(upscaled, original)
        
        if mode == "auto":
            # Auto-apply based on analysis
            if analysis.get("artifact_level", 0) > 0.3:
                processed = self._reduce_artifacts(processed, strength=0.5)
            
            if analysis.get("edge_sharpness", 0) < 0.6:
                processed = self._enhance_edges(processed, strength=0.4)
            
            if analysis.get("ringing", 0) > 0.2:
                processed = self._reduce_ringing(processed, strength=0.3)
            
            if analysis.get("color_consistency", 0) < 0.7:
                processed = self._improve_color_consistency(processed, original)
        
        elif mode == "aggressive":
            # Apply all enhancements
            processed = self._reduce_artifacts(processed, strength=0.7)
            processed = self._enhance_edges(processed, strength=0.6)
            processed = self._reduce_ringing(processed, strength=0.5)
            if original:
                processed = self._improve_color_consistency(processed, original)
        
        elif mode == "conservative":
            # Minimal enhancements
            if analysis.get("artifact_level", 0) > 0.5:
                processed = self._reduce_artifacts(processed, strength=0.3)
        
        return processed
    
    def _analyze_upscaled(
        self,
        upscaled: Image.Image,
        original: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """Analyze upscaled image quality."""
        upscaled_array = np.array(upscaled.convert("RGB"))
        
        # Artifact detection
        artifact_level = self._detect_artifacts(upscaled_array)
        
        # Edge sharpness
        edge_sharpness = self._measure_edge_sharpness(upscaled_array)
        
        # Ringing detection
        ringing = self._detect_ringing(upscaled_array)
        
        # Color consistency
        color_consistency = 1.0
        if original:
            color_consistency = self._measure_color_consistency(upscaled, original)
        
        return {
            "artifact_level": artifact_level,
            "edge_sharpness": edge_sharpness,
            "ringing": ringing,
            "color_consistency": color_consistency,
        }
    
    def _detect_artifacts(self, img_array: np.ndarray) -> float:
        """Detect artifacts (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.2  # Default
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect block artifacts (DCT-like)
        # High frequency patterns that shouldn't be there
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = np.abs(gray.astype(float) - blurred.astype(float))
        
        # Block patterns (simplified)
        block_score = np.std(diff) / 255.0
        
        return min(1.0, block_score)
    
    def _measure_edge_sharpness(self, img_array: np.ndarray) -> float:
        """Measure edge sharpness (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.7  # Default
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var() / 2000.0
        
        return min(1.0, max(0.0, sharpness))
    
    def _detect_ringing(self, img_array: np.ndarray) -> float:
        """Detect ringing artifacts (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.1  # Default
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Ringing appears as oscillations near edges
        # Detect high-frequency oscillations
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0
        
        # Check for oscillations near edges
        # Simplified: check variance in edge regions
        if np.sum(edge_mask) > 0:
            edge_region = gray[edge_mask]
            ringing_score = np.std(edge_region) / 255.0
        else:
            ringing_score = 0.0
        
        return min(1.0, ringing_score * 2.0)  # Amplify for detection
    
    def _measure_color_consistency(
        self,
        upscaled: Image.Image,
        original: Image.Image
    ) -> float:
        """Measure color consistency between original and upscaled."""
        # Resize original to match upscaled size for comparison
        orig_resized = original.resize(upscaled.size, Image.Resampling.LANCZOS)
        
        upscaled_array = np.array(upscaled.convert("RGB"), dtype=np.float32)
        orig_array = np.array(orig_resized.convert("RGB"), dtype=np.float32)
        
        # Calculate color difference
        diff = np.abs(upscaled_array - orig_array)
        mean_diff = np.mean(diff) / 255.0
        
        # Consistency is inverse of difference
        consistency = 1.0 - mean_diff
        
        return max(0.0, min(1.0, consistency))
    
    def _reduce_artifacts(self, image: Image.Image, strength: float = 0.5) -> Image.Image:
        """Reduce artifacts."""
        if not CV2_AVAILABLE:
            # PIL fallback
            return image.filter(ImageFilter.SMOOTH_MORE)
        
        img_array = np.array(image)
        
        # Bilateral filter to reduce artifacts while preserving edges
        filtered = cv2.bilateralFilter(
            img_array,
            d=9,
            sigmaColor=50 * strength,
            sigmaSpace=50 * strength
        )
        
        # Blend with original
        blended = cv2.addWeighted(
            img_array, 1.0 - strength * 0.5,
            filtered, strength * 0.5,
            0
        )
        
        return Image.fromarray(blended)
    
    def _enhance_edges(self, image: Image.Image, strength: float = 0.4) -> Image.Image:
        """Enhance edges."""
        if not CV2_AVAILABLE:
            # PIL fallback
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        img_array = np.array(image)
        
        # Unsharp mask for edge enhancement
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 1.5)
        sharpened = cv2.addWeighted(img_array, 1.0 + strength, gaussian, -strength, 0)
        
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
    
    def _reduce_ringing(self, image: Image.Image, strength: float = 0.3) -> Image.Image:
        """Reduce ringing artifacts."""
        if not CV2_AVAILABLE:
            return image
        
        img_array = np.array(image)
        
        # Gentle smoothing to reduce oscillations
        smoothed = cv2.GaussianBlur(img_array, (3, 3), 0.5)
        
        # Blend with original
        blended = cv2.addWeighted(
            img_array, 1.0 - strength,
            smoothed, strength,
            0
        )
        
        return Image.fromarray(blended)
    
    def _improve_color_consistency(
        self,
        upscaled: Image.Image,
        original: Image.Image
    ) -> Image.Image:
        """Improve color consistency with original."""
        # Match color statistics
        orig_resized = original.resize(upscaled.size, Image.Resampling.LANCZOS)
        
        upscaled_array = np.array(upscaled.convert("RGB"), dtype=np.float32)
        orig_array = np.array(orig_resized.convert("RGB"), dtype=np.float32)
        
        # Calculate mean and std for each channel
        upscaled_mean = np.mean(upscaled_array, axis=(0, 1))
        upscaled_std = np.std(upscaled_array, axis=(0, 1))
        
        orig_mean = np.mean(orig_array, axis=(0, 1))
        orig_std = np.std(orig_array, axis=(0, 1))
        
        # Match statistics
        for c in range(3):
            if upscaled_std[c] > 0:
                upscaled_array[:, :, c] = (
                    (upscaled_array[:, :, c] - upscaled_mean[c]) *
                    (orig_std[c] / upscaled_std[c]) +
                    orig_mean[c]
                )
        
        upscaled_array = np.clip(upscaled_array, 0, 255).astype(np.uint8)
        return Image.fromarray(upscaled_array)



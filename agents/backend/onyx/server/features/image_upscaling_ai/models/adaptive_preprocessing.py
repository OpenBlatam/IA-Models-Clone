"""
Adaptive Preprocessing
======================

Advanced preprocessing techniques that adapt to image characteristics.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import math

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class AdaptivePreprocessor:
    """
    Adaptive preprocessing that adjusts based on image analysis.
    
    Features:
    - Noise detection and reduction
    - Contrast enhancement
    - Sharpness adjustment
    - Color correction
    - Artifact removal
    - Adaptive filtering
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
    
    def preprocess(
        self,
        image: Image.Image,
        analysis: Optional[Dict[str, Any]] = None,
        mode: str = "auto"
    ) -> Image.Image:
        """
        Preprocess image adaptively.
        
        Args:
            image: Input image
            analysis: Image analysis results (auto-analyze if None)
            mode: Preprocessing mode ('auto', 'aggressive', 'conservative', 'none')
            
        Returns:
            Preprocessed image
        """
        if mode == "none":
            return image
        
        # Analyze image if needed
        if analysis is None:
            analysis = self._analyze_image(image)
        
        processed = image
        
        if mode == "auto":
            # Auto-select based on analysis
            if analysis.get("noise_level", 0) > 0.3:
                processed = self._denoise(processed, strength=0.5)
            
            if analysis.get("contrast", 0) < 0.4:
                processed = self._enhance_contrast(processed, factor=1.2)
            
            if analysis.get("sharpness", 0) < 0.5:
                processed = self._sharpen(processed, strength=0.3)
        
        elif mode == "aggressive":
            # Apply all enhancements
            processed = self._denoise(processed, strength=0.7)
            processed = self._enhance_contrast(processed, factor=1.3)
            processed = self._sharpen(processed, strength=0.5)
            processed = self._color_correct(processed)
        
        elif mode == "conservative":
            # Minimal enhancements
            if analysis.get("noise_level", 0) > 0.5:
                processed = self._denoise(processed, strength=0.3)
        
        return processed
    
    def _analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image characteristics."""
        img_array = np.array(image.convert("RGB"))
        
        # Calculate statistics
        mean = np.mean(img_array, axis=(0, 1))
        std = np.std(img_array, axis=(0, 1))
        
        # Contrast (standard deviation)
        contrast = np.mean(std) / 255.0
        
        # Brightness
        brightness = np.mean(mean) / 255.0
        
        # Noise estimation
        noise_level = self._estimate_noise(img_array)
        
        # Sharpness
        sharpness = self._estimate_sharpness(img_array)
        
        # Color balance
        color_balance = self._analyze_color_balance(img_array)
        
        return {
            "contrast": contrast,
            "brightness": brightness,
            "noise_level": noise_level,
            "sharpness": sharpness,
            "color_balance": color_balance,
        }
    
    def _estimate_noise(self, img_array: np.ndarray) -> float:
        """Estimate noise level (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.3  # Default
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # High-frequency content (noise)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = np.abs(gray.astype(float) - blurred.astype(float))
        noise = np.mean(diff) / 255.0
        
        return min(1.0, noise)
    
    def _estimate_sharpness(self, img_array: np.ndarray) -> float:
        """Estimate sharpness (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.5  # Default
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var() / 1000.0
        
        return min(1.0, max(0.0, sharpness))
    
    def _analyze_color_balance(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze color balance."""
        mean = np.mean(img_array, axis=(0, 1))
        gray = np.mean(mean)
        
        return {
            "r_balance": mean[0] / gray if gray > 0 else 1.0,
            "g_balance": mean[1] / gray if gray > 0 else 1.0,
            "b_balance": mean[2] / gray if gray > 0 else 1.0,
        }
    
    def _denoise(self, image: Image.Image, strength: float = 0.5) -> Image.Image:
        """Apply denoising."""
        if not CV2_AVAILABLE:
            # PIL fallback
            return image.filter(ImageFilter.SMOOTH_MORE)
        
        img_array = np.array(image)
        
        # Bilateral filter for denoising while preserving edges
        denoised = cv2.bilateralFilter(
            img_array,
            d=9,
            sigmaColor=75 * strength,
            sigmaSpace=75 * strength
        )
        
        return Image.fromarray(denoised)
    
    def _enhance_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _sharpen(self, image: Image.Image, strength: float = 0.3) -> Image.Image:
        """Apply sharpening."""
        # Unsharp mask
        if CV2_AVAILABLE:
            img_array = np.array(image)
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
            sharpened = cv2.addWeighted(img_array, 1.0 + strength, gaussian, -strength, 0)
            return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
        else:
            # PIL fallback
            return image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(strength * 100)))
    
    def _color_correct(self, image: Image.Image) -> Image.Image:
        """Apply color correction."""
        # Simple auto white balance
        img_array = np.array(image, dtype=np.float32)
        
        # Calculate channel means
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        
        # Gray world assumption
        gray = (r_mean + g_mean + b_mean) / 3.0
        
        # Adjust channels
        if r_mean > 0:
            img_array[:, :, 0] = img_array[:, :, 0] * (gray / r_mean)
        if g_mean > 0:
            img_array[:, :, 1] = img_array[:, :, 1] * (gray / g_mean)
        if b_mean > 0:
            img_array[:, :, 2] = img_array[:, :, 2] * (gray / b_mean)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)



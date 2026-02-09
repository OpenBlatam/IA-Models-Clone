"""
Image Processing Utilities
==========================

Utilities for image post-processing (anti-aliasing, artifact reduction, edge enhancement).
"""

import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageProcessingUtils:
    """Utilities for image post-processing."""
    
    @staticmethod
    def apply_anti_aliasing(
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """
        Apply anti-aliasing to reduce pixelation artifacts.
        
        Args:
            image: Input image
            strength: Anti-aliasing strength (0.0 to 1.0)
            
        Returns:
            Anti-aliased image
        """
        if strength <= 0:
            return image
        
        if not CV2_AVAILABLE:
            # Fallback to PIL-based anti-aliasing
            return image.filter(ImageFilter.SMOOTH_MORE)
        
        try:
            # Convert to numpy
            img_array = np.array(image, dtype=np.float32)
            
            # Apply Gaussian blur for anti-aliasing
            if len(img_array.shape) == 3:
                # Color image
                blurred = np.zeros_like(img_array)
                for i in range(3):
                    blurred[:, :, i] = cv2.GaussianBlur(
                        img_array[:, :, i],
                        (0, 0),
                        sigmaX=strength * 0.5
                    )
            else:
                # Grayscale
                blurred = cv2.GaussianBlur(img_array, (0, 0), sigmaX=strength * 0.5)
            
            # Blend original with blurred
            result = (1 - strength * 0.3) * img_array + (strength * 0.3) * blurred
            
            # Convert back to PIL
            result = np.clip(result, 0, 255).astype(np.uint8)
            return Image.fromarray(result)
            
        except Exception as e:
            logger.warning(f"Anti-aliasing failed: {e}")
            return image
    
    @staticmethod
    def reduce_artifacts(
        image: Image.Image,
        method: str = "bilateral"
    ) -> Image.Image:
        """
        Reduce upscaling artifacts.
        
        Args:
            image: Input image
            method: Method to use ('bilateral', 'median', 'gaussian')
            
        Returns:
            Processed image
        """
        if not CV2_AVAILABLE:
            # Fallback to PIL-based smoothing
            return image.filter(ImageFilter.SMOOTH)
        
        try:
            img_array = np.array(image)
            
            if method == "bilateral":
                # Bilateral filter preserves edges while reducing noise
                if len(img_array.shape) == 3:
                    filtered = cv2.bilateralFilter(
                        img_array,
                        d=9,
                        sigmaColor=75,
                        sigmaSpace=75
                    )
                else:
                    filtered = cv2.bilateralFilter(
                        img_array,
                        d=9,
                        sigmaColor=75,
                        sigmaSpace=75
                    )
            elif method == "median":
                # Median filter for noise reduction
                filtered = cv2.medianBlur(img_array, 3)
            elif method == "gaussian":
                # Gaussian blur
                filtered = cv2.GaussianBlur(img_array, (3, 3), 0)
            else:
                filtered = img_array
            
            return Image.fromarray(filtered)
            
        except Exception as e:
            logger.warning(f"Artifact reduction failed: {e}")
            return image
    
    @staticmethod
    def enhance_edges(
        image: Image.Image,
        strength: float = 1.2
    ) -> Image.Image:
        """
        Enhance edges for better sharpness.
        
        Args:
            image: Input image
            strength: Edge enhancement strength
            
        Returns:
            Enhanced image
        """
        # Apply unsharp mask
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(strength)
        
        return enhanced
    
    @staticmethod
    def enhance_with_frequency_analysis(
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """
        Enhance image using frequency domain analysis (FFT).
        
        Args:
            image: Input image
            strength: Enhancement strength (0.0-1.0)
            
        Returns:
            Enhanced image
        """
        try:
            img_array = np.array(image, dtype=np.float32)
            
            if len(img_array.shape) == 3:
                enhanced_channels = []
                for i in range(3):
                    channel = img_array[:, :, i]
                    # FFT
                    fft = np.fft.fft2(channel)
                    fft_shift = np.fft.fftshift(fft)
                    
                    # Enhance high frequencies
                    rows, cols = channel.shape
                    crow, ccol = rows // 2, cols // 2
                    
                    # Create high-pass filter
                    mask = np.ones((rows, cols), np.float32)
                    r = int(min(rows, cols) * 0.1 * (1 - strength))
                    center = [crow, ccol]
                    y, x = np.ogrid[:rows, :cols]
                    mask_area = (x - center[1])**2 + (y - center[0])**2 <= r*r
                    mask[mask_area] = 1.0 - strength * 0.3
                    
                    # Apply filter
                    fft_shift = fft_shift * mask
                    fft_ishift = np.fft.ifftshift(fft_shift)
                    enhanced = np.fft.ifft2(fft_ishift)
                    enhanced = np.real(enhanced)
                    
                    enhanced_channels.append(enhanced)
                
                enhanced_array = np.stack(enhanced_channels, axis=2)
            else:
                # Grayscale
                fft = np.fft.fft2(img_array)
                fft_shift = np.fft.fftshift(fft)
                
                rows, cols = img_array.shape
                crow, ccol = rows // 2, cols // 2
                
                mask = np.ones((rows, cols), np.float32)
                r = int(min(rows, cols) * 0.1 * (1 - strength))
                center = [crow, ccol]
                y, x = np.ogrid[:rows, :cols]
                mask_area = (x - center[1])**2 + (y - center[0])**2 <= r*r
                mask[mask_area] = 1.0 - strength * 0.3
                
                fft_shift = fft_shift * mask
                fft_ishift = np.fft.ifftshift(fft_shift)
                enhanced = np.fft.ifft2(fft_ishift)
                enhanced_array = np.real(enhanced)
            
            # Normalize and convert back
            enhanced_array = np.clip(enhanced_array, 0, 255).astype(np.uint8)
            return Image.fromarray(enhanced_array)
            
        except Exception as e:
            logger.warning(f"Frequency analysis enhancement failed: {e}")
            return image
    
    @staticmethod
    def adaptive_contrast_enhancement(
        image: Image.Image,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8)
    ) -> Image.Image:
        """
        Apply adaptive contrast enhancement (CLAHE).
        
        Args:
            image: Input image
            clip_limit: Contrast limiting threshold
            tile_grid_size: Size of grid for adaptive histogram equalization
            
        Returns:
            Enhanced image
        """
        if not CV2_AVAILABLE:
            # Fallback to PIL enhancement
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
        
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                # Color image
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                l = clahe.apply(l)
                
                # Merge channels
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(img_array)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            logger.warning(f"Adaptive contrast enhancement failed: {e}")
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
    
    @staticmethod
    def texture_enhancement(
        image: Image.Image,
        strength: float = 0.3
    ) -> Image.Image:
        """
        Enhance texture details in image.
        
        Args:
            image: Input image
            strength: Enhancement strength (0.0-1.0)
            
        Returns:
            Enhanced image
        """
        if not CV2_AVAILABLE:
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        try:
            img_array = np.array(image)
            
            # Apply unsharp mask for texture enhancement
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
            unsharp = cv2.addWeighted(img_array, 1.0 + strength, gaussian, -strength, 0)
            
            # Enhance edges
            if len(unsharp.shape) == 3:
                gray = cv2.cvtColor(unsharp, cv2.COLOR_RGB2GRAY)
            else:
                gray = unsharp
            
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) if len(unsharp.shape) == 3 else edges
            
            # Blend edges
            enhanced = cv2.addWeighted(
                unsharp,
                1.0 - strength * 0.2,
                edges_colored,
                strength * 0.2,
                0
            )
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            return Image.fromarray(enhanced)
            
        except Exception as e:
            logger.warning(f"Texture enhancement failed: {e}")
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    @staticmethod
    def color_enhancement(
        image: Image.Image,
        saturation: float = 1.1,
        vibrance: float = 1.05
    ) -> Image.Image:
        """
        Enhance color saturation and vibrance.
        
        Args:
            image: Input image
            saturation: Saturation multiplier
            vibrance: Vibrance multiplier (selective saturation)
            
        Returns:
            Enhanced image
        """
        # Saturation enhancement
        enhancer = ImageEnhance.Color(image)
        enhanced = enhancer.enhance(saturation)
        
        # Additional vibrance (selective saturation boost)
        if vibrance > 1.0 and CV2_AVAILABLE:
            try:
                img_array = np.array(enhanced)
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # Boost less saturated areas more
                saturation_channel = hsv[:, :, 1]
                mask = saturation_channel < 128  # Less saturated areas
                hsv[:, :, 1][mask] = np.clip(
                    hsv[:, :, 1][mask] * vibrance,
                    0, 255
                )
                
                hsv = hsv.astype(np.uint8)
                enhanced_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                enhanced = Image.fromarray(enhanced_array)
            except Exception:
                pass
        
        return enhanced


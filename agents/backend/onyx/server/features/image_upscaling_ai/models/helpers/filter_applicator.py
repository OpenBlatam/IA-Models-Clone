"""
Filter Applicator Utilities
===========================

Utilities for applying various image filters and enhancements.
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FilterApplicator:
    """Handles application of various image filters."""
    
    @staticmethod
    def apply_gaussian_blur(
        image_array: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        Apply Gaussian blur to image array.
        
        Args:
            image_array: Image as numpy array
            sigma: Blur sigma value
            
        Returns:
            Blurred image array
        """
        if len(image_array.shape) == 3:
            blurred = np.zeros_like(image_array)
            for i in range(3):
                blurred[:, :, i] = cv2.GaussianBlur(
                    image_array[:, :, i],
                    (0, 0),
                    sigmaX=sigma
                )
            return blurred
        else:
            return cv2.GaussianBlur(image_array, (0, 0), sigmaX=sigma)
    
    @staticmethod
    def apply_bilateral_filter(
        image_array: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter to preserve edges while reducing noise.
        
        Args:
            image_array: Image as numpy array
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            Filtered image array
        """
        return cv2.bilateralFilter(
            image_array,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )
    
    @staticmethod
    def apply_median_filter(
        image_array: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Apply median filter for noise reduction.
        
        Args:
            image_array: Image as numpy array
            kernel_size: Kernel size (must be odd)
            
        Returns:
            Filtered image array
        """
        return cv2.medianBlur(image_array, kernel_size)
    
    @staticmethod
    def apply_edge_preserving_filter(
        image_array: np.ndarray,
        flags: int = 1,
        sigma_s: float = 50,
        sigma_r: float = 0.4
    ) -> np.ndarray:
        """
        Apply edge-preserving filter.
        
        Args:
            image_array: Image as numpy array
            flags: Filter flags
            sigma_s: Spatial sigma
            sigma_r: Range sigma
            
        Returns:
            Filtered image array
        """
        return cv2.edgePreservingFilter(
            image_array,
            flags=flags,
            sigma_s=sigma_s,
            sigma_r=sigma_r
        )
    
    @staticmethod
    def apply_unsharp_mask(
        image: Image.Image,
        radius: int = 1,
        percent: int = 150,
        threshold: int = 3
    ) -> Image.Image:
        """
        Apply unsharp mask for sharpness enhancement.
        
        Args:
            image: PIL Image
            radius: Radius of the blur
            percent: Enhancement percentage
            threshold: Threshold for enhancement
            
        Returns:
            Enhanced PIL Image
        """
        return image.filter(ImageFilter.UnsharpMask(
            radius=radius,
            percent=percent,
            threshold=threshold
        ))
    
    @staticmethod
    def apply_sharpness_enhancement(
        image: Image.Image,
        strength: float = 1.2
    ) -> Image.Image:
        """
        Apply sharpness enhancement.
        
        Args:
            image: PIL Image
            strength: Enhancement strength
            
        Returns:
            Enhanced PIL Image
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(strength)
    
    @staticmethod
    def blend_images(
        original: np.ndarray,
        processed: np.ndarray,
        blend_factor: float
    ) -> np.ndarray:
        """
        Blend original and processed images.
        
        Args:
            original: Original image array
            processed: Processed image array
            blend_factor: Blend factor (0.0 to 1.0)
            
        Returns:
            Blended image array
        """
        result = (1 - blend_factor) * original + blend_factor * processed
        return np.clip(result, 0, 255).astype(np.uint8)



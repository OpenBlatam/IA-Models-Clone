"""
Image Enhancer Helper
=====================

Enhances image quality before processing.
"""

import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Enhances image quality for better processing."""
    
    @staticmethod
    def enhance_contrast(
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """
        Enhance image contrast.
        
        Args:
            image: PIL Image
            factor: Enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_sharpness(
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """
        Enhance image sharpness.
        
        Args:
            image: PIL Image
            factor: Enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_brightness(
        image: Image.Image,
        factor: float = 1.1
    ) -> Image.Image:
        """
        Enhance image brightness.
        
        Args:
            image: PIL Image
            factor: Enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_color(
        image: Image.Image,
        factor: float = 1.1
    ) -> Image.Image:
        """
        Enhance image color saturation.
        
        Args:
            image: PIL Image
            factor: Enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def reduce_noise(
        image: Image.Image,
        radius: float = 1.0
    ) -> Image.Image:
        """
        Reduce image noise.
        
        Args:
            image: PIL Image
            radius: Filter radius
            
        Returns:
            Denoised image
        """
        return image.filter(ImageFilter.MedianFilter(size=int(radius * 2) + 1))
    
    @staticmethod
    def auto_enhance(
        image: Image.Image,
        enhance_contrast: bool = True,
        enhance_sharpness: bool = True,
        enhance_brightness: bool = False,
        reduce_noise: bool = False
    ) -> Image.Image:
        """
        Automatically enhance image with multiple techniques.
        
        Args:
            image: PIL Image
            enhance_contrast: Apply contrast enhancement
            enhance_sharpness: Apply sharpness enhancement
            enhance_brightness: Apply brightness enhancement
            reduce_noise: Apply noise reduction
            
        Returns:
            Enhanced image
        """
        enhanced = image
        
        if reduce_noise:
            enhanced = ImageEnhancer.reduce_noise(enhanced)
        
        if enhance_contrast:
            enhanced = ImageEnhancer.enhance_contrast(enhanced)
        
        if enhance_sharpness:
            enhanced = ImageEnhancer.enhance_sharpness(enhanced)
        
        if enhance_brightness:
            enhanced = ImageEnhancer.enhance_brightness(enhanced)
        
        return enhanced



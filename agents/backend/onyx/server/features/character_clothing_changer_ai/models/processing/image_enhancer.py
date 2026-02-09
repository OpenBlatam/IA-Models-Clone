"""
Image Enhancer for Flux2 Clothing Changer
==========================================

Enhances images automatically to improve clothing change results.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Enhances images for better clothing change results."""
    
    def __init__(
        self,
        enhance_brightness: bool = True,
        enhance_contrast: bool = True,
        enhance_sharpness: bool = True,
        enhance_color: bool = False,
        brightness_factor: float = 1.1,
        contrast_factor: float = 1.15,
        sharpness_factor: float = 1.2,
        color_factor: float = 1.1,
        auto_tune: bool = True,
    ):
        """
        Initialize image enhancer.
        
        Args:
            enhance_brightness: Whether to enhance brightness
            enhance_contrast: Whether to enhance contrast
            enhance_sharpness: Whether to enhance sharpness
            enhance_color: Whether to enhance color saturation
            brightness_factor: Brightness enhancement factor
            contrast_factor: Contrast enhancement factor
            sharpness_factor: Sharpness enhancement factor
            color_factor: Color saturation enhancement factor
            auto_tune: Automatically adjust factors based on image analysis
        """
        self.enhance_brightness = enhance_brightness
        self.enhance_contrast = enhance_contrast
        self.enhance_sharpness = enhance_sharpness
        self.enhance_color = enhance_color
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor
        self.color_factor = color_factor
        self.auto_tune = auto_tune
    
    def enhance(
        self,
        image: Image.Image,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Enhance image for better processing.
        
        Args:
            image: Image to enhance
            metrics: Optional image metrics from validator
            
        Returns:
            Tuple of (enhanced_image, enhancement_info)
        """
        enhancement_info = {
            "enhanced": False,
            "operations": [],
            "factors": {},
        }
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            enhancement_info["operations"].append("converted_to_rgb")
        
        # Auto-tune factors based on image analysis
        if self.auto_tune and metrics:
            brightness = metrics.get("brightness", 0.5)
            contrast = metrics.get("contrast", 0.5)
            sharpness = metrics.get("sharpness", 0.5)
            
            # Adjust brightness factor
            if brightness < 0.4:
                brightness_factor = 1.2
            elif brightness > 0.7:
                brightness_factor = 1.05
            else:
                brightness_factor = self.brightness_factor
            
            # Adjust contrast factor
            if contrast < 0.3:
                contrast_factor = 1.25
            elif contrast > 0.6:
                contrast_factor = 1.1
            else:
                contrast_factor = self.contrast_factor
            
            # Adjust sharpness factor
            if sharpness is not None and sharpness < 0.3:
                sharpness_factor = 1.3
            elif sharpness is not None and sharpness > 0.7:
                sharpness_factor = 1.1
            else:
                sharpness_factor = self.sharpness_factor
        else:
            brightness_factor = self.brightness_factor
            contrast_factor = self.contrast_factor
            sharpness_factor = self.sharpness_factor
        
        # Apply enhancements
        if self.enhance_brightness:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
            enhancement_info["operations"].append("brightness")
            enhancement_info["factors"]["brightness"] = brightness_factor
        
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
            enhancement_info["operations"].append("contrast")
            enhancement_info["factors"]["contrast"] = contrast_factor
        
        if self.enhance_sharpness:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness_factor)
            enhancement_info["operations"].append("sharpness")
            enhancement_info["factors"]["sharpness"] = sharpness_factor
        
        if self.enhance_color:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(self.color_factor)
            enhancement_info["operations"].append("color")
            enhancement_info["factors"]["color"] = self.color_factor
        
        enhancement_info["enhanced"] = len(enhancement_info["operations"]) > 0
        
        if enhancement_info["enhanced"]:
            logger.info(f"Image enhanced with operations: {enhancement_info['operations']}")
        
        return image, enhancement_info
    
    def quick_enhance(self, image: Image.Image) -> Image.Image:
        """
        Quick enhancement with default settings.
        
        Args:
            image: Image to enhance
            
        Returns:
            Enhanced image
        """
        enhanced, _ = self.enhance(image)
        return enhanced
    
    # Static methods for backward compatibility with helpers
    @staticmethod
    def enhance_contrast(
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_sharpness(
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """Enhance image sharpness."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_brightness(
        image: Image.Image,
        factor: float = 1.1
    ) -> Image.Image:
        """Enhance image brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_color(
        image: Image.Image,
        factor: float = 1.1
    ) -> Image.Image:
        """Enhance image color saturation."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def reduce_noise(
        image: Image.Image,
        radius: float = 1.0
    ) -> Image.Image:
        """Reduce image noise."""
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



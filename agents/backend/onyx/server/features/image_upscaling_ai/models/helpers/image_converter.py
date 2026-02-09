"""
Image Converter Utilities
=========================

Utilities for converting between PIL and OpenCV image formats.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Union
import logging

logger = logging.getLogger(__name__)


class ImageConverter:
    """Handles conversion between PIL and OpenCV image formats."""
    
    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format (BGR).
        
        Args:
            image: PIL Image
            
        Returns:
            OpenCV image array (BGR format)
        """
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    
    @staticmethod
    def cv2_to_pil(image_cv: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image (BGR) to PIL Image (RGB).
        
        Args:
            image_cv: OpenCV image array (BGR format)
            
        Returns:
            PIL Image (RGB format)
        """
        if len(image_cv.shape) == 3:
            image_array = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        else:
            image_array = image_cv
        return Image.fromarray(image_array)
    
    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """
        Ensure image is in RGB mode.
        
        Args:
            image: PIL Image
            
        Returns:
            RGB PIL Image
        """
        if image.mode != "RGB":
            return image.convert("RGB")
        return image



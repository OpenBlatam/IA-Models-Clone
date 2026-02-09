"""
Mask Generator
==============

Generates masks for clothing regions using various methods.
"""

import logging
from typing import Optional
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..constants import CLOTHING_REGION_RATIO

logger = logging.getLogger(__name__)


class MaskGenerator:
    """Generates masks for clothing regions using various methods."""
    
    @staticmethod
    def generate_simple_mask(image: Image.Image, region_ratio: float = CLOTHING_REGION_RATIO) -> Image.Image:
        """
        Generate a simple mask covering lower body region.
        
        Args:
            image: Input image
            region_ratio: Ratio of image height to mask (default: 0.6 for lower 60%)
            
        Returns:
            Mask image (L mode)
        """
        mask = Image.new("L", image.size, 0)
        mask_array = np.array(mask)
        h, w = mask_array.shape
        mask_array[int(h * (1 - region_ratio)):, :] = 255
        return Image.fromarray(mask_array)
    
    @staticmethod
    def generate_smart_mask(image: Image.Image) -> Image.Image:
        """
        Generate a smarter mask using edge detection and color analysis.
        
        Args:
            image: Input image
            
        Returns:
            Mask image (L mode)
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, using simple mask")
            return MaskGenerator.generate_simple_mask(image)
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges to create regions
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Focus on lower region (clothing area)
            h, w = dilated.shape
            lower_region = int(h * (1 - CLOTHING_REGION_RATIO))
            
            # Create mask focusing on lower body
            mask = np.zeros_like(dilated)
            mask[lower_region:, :] = dilated[lower_region:, :]
            
            # Fill holes and smooth
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Threshold to binary
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
            
            # Ensure we have a reasonable mask size
            if np.sum(mask) < (h * w * 0.1):  # If mask is too small
                # Fallback to simple mask
                mask[lower_region:, :] = 255
            
            return Image.fromarray(mask)
            
        except Exception as e:
            logger.warning(f"Error in smart mask generation: {e}, falling back to simple mask")
            return MaskGenerator.generate_simple_mask(image)
    
    @staticmethod
    def refine_mask(mask: Image.Image, image: Image.Image) -> Image.Image:
        """
        Refine mask using image information.
        
        Args:
            mask: Initial mask
            image: Original image
            
        Returns:
            Refined mask
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, returning original mask")
            return mask
        
        try:
            mask_array = np.array(mask)
            img_array = np.array(image)
            
            # Apply morphological operations
            kernel = np.ones((7, 7), np.uint8)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
            
            # Smooth edges
            mask_array = cv2.GaussianBlur(mask_array, (9, 9), 0)
            
            return Image.fromarray(mask_array)
        except Exception as e:
            logger.warning(f"Error refining mask: {e}")
            return mask



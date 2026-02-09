"""
Dimension Calculator Utilities
===============================

Utilities for calculating image dimensions and scale factors.
"""

from typing import Tuple
from PIL import Image


class DimensionCalculator:
    """Handles dimension calculations for image upscaling."""
    
    @staticmethod
    def calculate_new_dimensions(
        width: int,
        height: int,
        scale_factor: float
    ) -> Tuple[int, int]:
        """
        Calculate new dimensions based on scale factor.
        
        Args:
            width: Original width
            height: Original height
            scale_factor: Scale factor to apply
            
        Returns:
            Tuple of (new_width, new_height)
        """
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return new_width, new_height
    
    @staticmethod
    def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (width, height)
        """
        return image.size
    
    @staticmethod
    def calculate_scale_for_pass(
        remaining_scale: float,
        max_pass_scale: float = 2.0
    ) -> float:
        """
        Calculate scale factor for a single upscaling pass.
        
        Args:
            remaining_scale: Remaining scale to apply
            max_pass_scale: Maximum scale per pass
            
        Returns:
            Scale factor for this pass
        """
        if remaining_scale >= max_pass_scale:
            return max_pass_scale
        return remaining_scale



"""
Image Optimizer Helper
======================

Optimizes images for processing efficiency.
"""

import logging
from PIL import Image
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ImageOptimizer:
    """Optimizes images for processing."""
    
    @staticmethod
    def optimize_for_processing(
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect: bool = True,
        quality: int = 95
    ) -> Image.Image:
        """
        Optimize image for processing.
        
        Args:
            image: PIL Image
            target_size: Target size (width, height)
            maintain_aspect: Maintain aspect ratio
            quality: Quality for optimization
            
        Returns:
            Optimized image
        """
        optimized = image
        
        if target_size:
            if maintain_aspect:
                optimized = ImageOptimizer._resize_maintain_aspect(
                    optimized, target_size
                )
            else:
                optimized = optimized.resize(target_size, Image.Resampling.LANCZOS)
        
        return optimized
    
    @staticmethod
    def _resize_maintain_aspect(
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize maintaining aspect ratio."""
        target_width, target_height = target_size
        width, height = image.size
        
        # Calculate aspect ratios
        image_aspect = width / height
        target_aspect = target_width / target_height
        
        if image_aspect > target_aspect:
            # Image is wider
            new_width = target_width
            new_height = int(target_width / image_aspect)
        else:
            # Image is taller
            new_height = target_height
            new_width = int(target_height * image_aspect)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def optimize_memory(
        image: Image.Image,
        max_dimension: int = 1024
    ) -> Image.Image:
        """
        Optimize image for memory usage.
        
        Args:
            image: PIL Image
            max_dimension: Maximum dimension
            
        Returns:
            Memory-optimized image
        """
        width, height = image.size
        
        if max(width, height) > max_dimension:
            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def optimize_for_quality(
        image: Image.Image,
        min_dimension: int = 512
    ) -> Image.Image:
        """
        Optimize image for quality.
        
        Args:
            image: PIL Image
            min_dimension: Minimum dimension
            
        Returns:
            Quality-optimized image
        """
        width, height = image.size
        
        if min(width, height) < min_dimension:
            ratio = min_dimension / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image



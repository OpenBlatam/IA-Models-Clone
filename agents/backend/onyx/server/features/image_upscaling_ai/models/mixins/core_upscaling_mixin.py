"""
Core Upscaling Mixin

Contains the core upscaling methods and basic functionality.
This mixin should be inherited by the main AdvancedUpscaling class.
"""

import logging
import time
from typing import Union, Tuple, Optional, Callable
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityMetrics,
    ImageQualityValidator,
    QualityCalculator,
    MethodSelector,
    DimensionCalculator,
    UpscalingAlgorithms,
)

logger = logging.getLogger(__name__)


class CoreUpscalingMixin:
    """
    Mixin providing core upscaling functionality.
    
    This mixin contains:
    - Basic upscaling methods (lanczos, bicubic, opencv)
    - Main upscale() method
    - Multi-scale upscaling
    - Adaptive upscaling
    - Retry functionality
    """
    
    def upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        use_cache: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_metrics: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale image with selected method and optional caching.
        
        This is the main upscaling method that should be implemented
        in the main class with full caching, validation, and metrics support.
        
        Args:
            image: Input image (PIL Image or path)
            scale_factor: Scale factor
            method: Method to use ('lanczos', 'bicubic', 'opencv', 'multi_scale')
            use_cache: Override cache setting
            progress_callback: Optional callback(current, total) for progress
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        # This method should be implemented in the main class
        # as it requires access to self.cache, self.stats, etc.
        raise NotImplementedError("This method should be implemented in the main class")
    
    @staticmethod
    def upscale_lanczos(
        image: Image.Image,
        scale_factor: float,
        taps: int = 3
    ) -> Image.Image:
        """
        Upscale using Lanczos resampling with configurable taps.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            taps: Number of taps (3 for Lanczos3, 2 for Lanczos2)
            
        Returns:
            Upscaled image
        """
        width, height = DimensionCalculator.get_image_dimensions(image)
        new_width, new_height = DimensionCalculator.calculate_new_dimensions(
            width, height, scale_factor
        )
        
        # Use Lanczos resampling
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def upscale_bicubic_enhanced(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Enhanced bicubic upscaling with post-processing.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        from ..helpers import FilterApplicator
        
        width, height = DimensionCalculator.get_image_dimensions(image)
        new_width, new_height = DimensionCalculator.calculate_new_dimensions(
            width, height, scale_factor
        )
        
        # Upscale with bicubic
        upscaled = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # Apply unsharp mask for sharpness
        upscaled = FilterApplicator.apply_unsharp_mask(upscaled, radius=1, percent=150, threshold=3)
        
        return upscaled
    
    @staticmethod
    def upscale_opencv_edsr(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Upscale using OpenCV EDSR-like super-resolution.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("OpenCV not available, falling back to Lanczos")
            return CoreUpscalingMixin.upscale_lanczos(image, scale_factor)
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        width, height = DimensionCalculator.get_image_dimensions(image)
        new_width, new_height = DimensionCalculator.calculate_new_dimensions(
            width, height, scale_factor
        )
        
        # Use EDSR-like upscaling (simulated with cv2.INTER_CUBIC)
        upscaled = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to PIL
        if len(upscaled.shape) == 3:
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        return Image.fromarray(upscaled)
    
    @staticmethod
    def multi_scale_upscale(
        image: Image.Image,
        scale_factor: float,
        steps: int = 2
    ) -> Image.Image:
        """
        Multi-scale upscaling for better quality at high scale factors.
        
        Args:
            image: Input image
            scale_factor: Total scale factor
            steps: Number of upscaling steps
            
        Returns:
            Upscaled image
        """
        if scale_factor <= 1.0:
            return image
        
        # Calculate scale per step
        scale_per_step = scale_factor ** (1.0 / steps)
        result = image
        
        for step in range(steps):
            result = CoreUpscalingMixin.upscale_lanczos(result, scale_per_step)
        
        return result
    
    @staticmethod
    def upscale_adaptive(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Adaptive upscaling that selects best method automatically.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        return UpscalingAlgorithms.upscale_adaptive(image, scale_factor)



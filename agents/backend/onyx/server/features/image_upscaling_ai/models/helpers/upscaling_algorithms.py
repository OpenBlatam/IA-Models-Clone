"""
Upscaling Algorithms
====================

Various upscaling algorithms and techniques.
"""

import logging
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class UpscalingAlgorithms:
    """Collection of upscaling algorithms."""
    
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
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Use Lanczos resampling (taps parameter is informational, PIL uses LANCZOS)
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
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Upscale with bicubic
        upscaled = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # Apply unsharp mask for sharpness
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        return upscaled
    
    @staticmethod
    def upscale_opencv_edsr(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Upscale using OpenCV's EDSR-like super resolution.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, falling back to Lanczos")
            return UpscalingAlgorithms.upscale_lanczos(image, scale_factor)
        
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            width, height = image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Use INTER_CUBIC for better quality
            upscaled_cv = cv2.resize(
                img_cv,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Apply edge-preserving filter
            upscaled_cv = cv2.edgePreservingFilter(upscaled_cv, flags=1, sigma_s=50, sigma_r=0.4)
            
            # Convert back to PIL
            if len(upscaled_cv.shape) == 3:
                upscaled_array = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB)
            else:
                upscaled_array = upscaled_cv
            
            return Image.fromarray(upscaled_array)
            
        except Exception as e:
            logger.warning(f"OpenCV upscaling failed: {e}, falling back to Lanczos")
            return UpscalingAlgorithms.upscale_lanczos(image, scale_factor)
    
    @staticmethod
    def multi_scale_upscale(
        image: Image.Image,
        scale_factor: float,
        passes: int = 2
    ) -> Image.Image:
        """
        Multi-scale upscaling for better quality.
        
        Args:
            image: Input image
            scale_factor: Total scale factor
            passes: Number of upscaling passes
            
        Returns:
            Upscaled image
        """
        from .image_processing_utils import ImageProcessingUtils
        
        current_image = image
        remaining_scale = scale_factor
        
        for i in range(passes):
            if remaining_scale <= 1.0:
                break
            
            # Calculate scale for this pass
            if remaining_scale >= 2.0:
                pass_scale = 2.0
            else:
                pass_scale = remaining_scale
            
            # Upscale
            current_image = UpscalingAlgorithms.upscale_lanczos(
                current_image,
                pass_scale,
                taps=3
            )
            
            # Apply anti-aliasing between passes
            if i < passes - 1:
                current_image = ImageProcessingUtils.apply_anti_aliasing(
                    current_image,
                    strength=0.3
                )
            
            remaining_scale /= pass_scale
        
        return current_image
    
    @staticmethod
    def upscale_adaptive(
        image: Image.Image,
        scale_factor: float,
        quality_threshold: float = 0.7
    ) -> Image.Image:
        """
        Adaptive upscaling that adjusts method based on image characteristics.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            quality_threshold: Quality threshold for method selection
            
        Returns:
            Upscaled image
        """
        from .quality_calculator_utils import QualityCalculator
        from .image_processing_utils import ImageProcessingUtils
        
        # Analyze image first
        quality = QualityCalculator.calculate_quality_metrics(image)
        
        # Select method based on quality
        if quality.overall_quality < quality_threshold:
            # Low quality - use multi-scale for better results
            return UpscalingAlgorithms.multi_scale_upscale(image, scale_factor, passes=3)
        elif scale_factor > 3.0:
            # Large scale - use multi-scale
            return UpscalingAlgorithms.multi_scale_upscale(image, scale_factor, passes=2)
        elif quality.noise_level > 15:
            # High noise - use opencv with denoising
            result = UpscalingAlgorithms.upscale_opencv_edsr(image, scale_factor)
            return ImageProcessingUtils.reduce_artifacts(result, method="bilateral")
        else:
            # Good quality - use enhanced bicubic
            return UpscalingAlgorithms.upscale_bicubic_enhanced(image, scale_factor)
    
    @staticmethod
    def upscale_esrgan_like(
        image: Image.Image,
        scale_factor: float,
        iterations: int = 2
    ) -> Image.Image:
        """
        ESRGAN-like upscaling using iterative enhancement.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            iterations: Number of enhancement iterations
            
        Returns:
            Upscaled image
        """
        from PIL import ImageFilter
        from .image_processing_utils import ImageProcessingUtils
        
        # Initial upscale with lanczos
        result = UpscalingAlgorithms.upscale_lanczos(image, scale_factor)
        
        # Iterative enhancement
        for i in range(iterations):
            # Apply edge enhancement
            result = ImageProcessingUtils.enhance_edges(result, strength=1.1)
            
            # Reduce artifacts
            result = ImageProcessingUtils.reduce_artifacts(result, method="bilateral")
            
            # Apply subtle sharpening
            if i < iterations - 1:
                result = result.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2))
        
        return result
    
    @staticmethod
    def upscale_waifu2x_like(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Waifu2x-like upscaling with noise reduction and upscaling.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, falling back to Lanczos")
            return UpscalingAlgorithms.upscale_lanczos(image, scale_factor)
        
        try:
            # Convert to numpy
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Denoise first (Waifu2x approach)
            denoised = cv2.fastNlMeansDenoisingColored(
                img_cv,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            # Upscale
            width, height = image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            upscaled = cv2.resize(
                denoised,
                (new_width, new_height),
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Convert back to PIL
            if len(upscaled.shape) == 3:
                upscaled_array = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
            else:
                upscaled_array = upscaled
            
            return Image.fromarray(upscaled_array)
            
        except Exception as e:
            logger.warning(f"Waifu2x-like upscaling failed: {e}, falling back to Lanczos")
            return UpscalingAlgorithms.upscale_lanczos(image, scale_factor)
    
    @staticmethod
    def upscale_real_esrgan_like(
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Real-ESRGAN-like upscaling with advanced processing.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        from .image_processing_utils import ImageProcessingUtils
        
        # Multi-scale approach
        if scale_factor > 2.0:
            # Use multi-scale for large factors
            result = UpscalingAlgorithms.multi_scale_upscale(image, scale_factor, passes=3)
        else:
            # Single pass with enhanced processing
            result = UpscalingAlgorithms.upscale_lanczos(image, scale_factor)
        
        # Advanced post-processing
        # 1. Edge-preserving denoising
        result = ImageProcessingUtils.reduce_artifacts(result, method="bilateral")
        
        # 2. Adaptive sharpening
        result = ImageProcessingUtils.enhance_edges(result, strength=1.15)
        
        # 3. Subtle anti-aliasing
        result = ImageProcessingUtils.apply_anti_aliasing(result, strength=0.2)
        
        return result


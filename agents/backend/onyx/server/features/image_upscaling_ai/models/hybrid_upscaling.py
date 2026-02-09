"""
Hybrid Upscaling
================

Combine multiple upscaling methods for optimal results.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .advanced_upscaling import AdvancedUpscaling
    from .realesrgan_integration import REALESRGAN_AVAILABLE, RealESRGANUpscaler
except ImportError:
    REALESRGAN_AVAILABLE = False
    AdvancedUpscaling = None
    RealESRGANUpscaler = None


class HybridUpscaler:
    """
    Hybrid upscaling that combines multiple methods.
    
    Features:
    - Method combination (e.g., Real-ESRGAN + post-processing)
    - Adaptive method selection
    - Quality-based method switching
    - Multi-stage upscaling
    """
    
    def __init__(
        self,
        primary_method: str = "realesrgan",
        secondary_method: str = "lanczos",
        blend_ratio: float = 0.7,
    ):
        """
        Initialize hybrid upscaler.
        
        Args:
            primary_method: Primary upscaling method
            secondary_method: Secondary method for blending
            blend_ratio: Ratio for blending (0.0-1.0, higher = more primary)
        """
        self.primary_method = primary_method
        self.secondary_method = secondary_method
        self.blend_ratio = blend_ratio
    
    def upscale_hybrid(
        self,
        image: Image.Image,
        scale_factor: float,
        method: str = "auto"
    ) -> Image.Image:
        """
        Upscale using hybrid approach.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method: Hybrid method ('auto', 'blend', 'two_stage', 'adaptive')
            
        Returns:
            Upscaled image
        """
        if method == "auto":
            method = self._select_hybrid_method(image, scale_factor)
        
        if method == "blend":
            return self._blend_methods(image, scale_factor)
        elif method == "two_stage":
            return self._two_stage_upscale(image, scale_factor)
        elif method == "adaptive":
            return self._adaptive_hybrid(image, scale_factor)
        else:
            return self._blend_methods(image, scale_factor)
    
    def _blend_methods(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Blend results from two methods."""
        # Upscale with primary method
        primary = self._upscale_with_method(image, scale_factor, self.primary_method)
        
        # Upscale with secondary method
        secondary = self._upscale_with_method(image, scale_factor, self.secondary_method)
        
        # Blend
        primary_array = np.array(primary, dtype=np.float32)
        secondary_array = np.array(secondary, dtype=np.float32)
        
        blended = (
            primary_array * self.blend_ratio +
            secondary_array * (1.0 - self.blend_ratio)
        ).astype(np.uint8)
        
        return Image.fromarray(blended)
    
    def _two_stage_upscale(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Two-stage upscaling: coarse then refine."""
        # Stage 1: Coarse upscaling (faster method)
        if scale_factor > 2.0:
            stage1_scale = 2.0
            stage1_result = self._upscale_with_method(
                image, stage1_scale, self.secondary_method
            )
        else:
            stage1_result = image
            stage1_scale = 1.0
        
        # Stage 2: Refine with high-quality method
        remaining_scale = scale_factor / stage1_scale
        if remaining_scale > 1.0:
            final = self._upscale_with_method(
                stage1_result, remaining_scale, self.primary_method
            )
        else:
            final = stage1_result
        
        return final
    
    def _adaptive_hybrid(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Adaptive hybrid: use different methods for different regions."""
        # Analyze image regions
        regions = self._analyze_regions(image)
        
        # Upscale each region with appropriate method
        upscaled_regions = []
        for region_type, region_image in regions:
            if region_type == "smooth":
                # Use fast method for smooth regions
                method = self.secondary_method
            else:
                # Use high-quality method for detailed regions
                method = self.primary_method
            
            upscaled = self._upscale_with_method(region_image, scale_factor, method)
            upscaled_regions.append((region_type, upscaled))
        
        # Combine regions (simplified - would need proper masking)
        # For now, just use primary method
        return self._upscale_with_method(image, scale_factor, self.primary_method)
    
    def _upscale_with_method(
        self,
        image: Image.Image,
        scale_factor: float,
        method: str
    ) -> Image.Image:
        """Upscale using specified method."""
        if method == "realesrgan" and REALESRGAN_AVAILABLE:
            try:
                upscaler = RealESRGANUpscaler()
                return upscaler.upscale(image, scale_factor)
            except Exception as e:
                logger.warning(f"Real-ESRGAN failed: {e}, falling back")
                method = "lanczos"
        
        if AdvancedUpscaling:
            if method == "lanczos":
                return AdvancedUpscaling.upscale_lanczos(image, scale_factor)
            elif method == "opencv_edsr":
                return AdvancedUpscaling.upscale_opencv_edsr(image, scale_factor)
            elif method == "bicubic_enhanced":
                return AdvancedUpscaling.upscale_bicubic_enhanced(image, scale_factor)
        
        # Fallback
        width, height = image.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def _select_hybrid_method(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> str:
        """Select best hybrid method."""
        # For high scale factors, use two-stage
        if scale_factor > 4.0:
            return "two_stage"
        
        # For medium scale, use blend
        if scale_factor > 2.0:
            return "blend"
        
        # For low scale, use single method
        return "blend"
    
    def _analyze_regions(
        self,
        image: Image.Image
    ) -> List[Tuple[str, Image.Image]]:
        """Analyze image regions (simplified)."""
        # This would use edge detection, texture analysis, etc.
        # For now, return whole image as "detailed"
        return [("detailed", image)]



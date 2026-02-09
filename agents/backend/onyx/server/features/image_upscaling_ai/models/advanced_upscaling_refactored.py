"""
Advanced Upscaling - Refactored Version
========================================

This is a refactored version using mixins for better modularity and maintainability.
The original advanced_upscaling.py is kept for backward compatibility.
"""

import logging
from typing import Tuple, Optional, Dict, Any, List, Callable, Union
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Import mixins
from .mixins import (
    CoreUpscalingMixin,
    EnhancementMixin,
    MLAIMixin,
    AnalysisMixin,
    PipelineMixin,
    AdvancedMethodsMixin,
    BatchProcessingMixin,
    CacheManagementMixin,
    OptimizationMixin,
    QualityAssuranceMixin,
    UtilityMixin,
)

# Import helpers
from .helpers import (
    UpscalingMetrics,
    QualityMetrics,
    ImageQualityValidator,
    QualityCalculator,
    MethodSelector,
    StatisticsManager,
    UpscalingCache,
    retry_on_failure,
)

# Import algorithms and postprocessing
from .advanced_upscaling_algorithms import UpscalingAlgorithmsStatic
from .advanced_upscaling_postprocessing import PostprocessingMethods

logger = logging.getLogger(__name__)

# Try to import Real-ESRGAN
try:
    from .realesrgan_integration import RealESRGANUpscaler, REALESRGAN_AVAILABLE
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANUpscaler = None

if not REALESRGAN_AVAILABLE:
    logger.warning("Real-ESRGAN not available. Install with: pip install realesrgan basicsr")


class AdvancedUpscalingRefactored(
    CoreUpscalingMixin,
    EnhancementMixin,
    MLAIMixin,
    AnalysisMixin,
    PipelineMixin,
    AdvancedMethodsMixin,
    BatchProcessingMixin,
    CacheManagementMixin,
    OptimizationMixin,
    QualityAssuranceMixin,
    UtilityMixin
):
    """
    Advanced upscaling techniques - refactored version using mixins.
    
    This class combines all mixins to provide a complete upscaling solution.
    The mixins provide:
    - Core upscaling functionality
    - Image enhancement
    - ML/AI methods
    - Analysis and reporting
    - Pipeline management
    - Advanced methods
    - Batch processing
    - Cache management
    - Optimization
    - Quality assurance
    - Utilities
    
    Features:
    - Multiple upscaling algorithms
    - Anti-aliasing
    - Artifact reduction
    - Edge preservation
    - Noise reduction
    - Caching system
    - Async processing
    - Quality metrics
    - Progress tracking
    - Batch processing
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 64,
        cache_ttl: int = 3600,
        max_workers: int = 4,
        validate_images: bool = True,
        enhance_images: bool = False,
        auto_select_method: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize Advanced Upscaling.
        
        Args:
            enable_cache: Enable result caching
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
            max_workers: Maximum worker threads
            validate_images: Validate images before processing
            enhance_images: Automatically enhance images
            auto_select_method: Automatically select best method
            max_retries: Maximum retry attempts
        """
        # Initialize core attributes
        self.enable_cache = enable_cache
        self.cache = UpscalingCache(max_size=cache_size, ttl=cache_ttl) if enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.validate_images = validate_images
        self.enhance_images = enhance_images
        self.auto_select_method = auto_select_method
        self.max_retries = max_retries
        self.stats = StatisticsManager.create_default_stats()
        
        logger.info(
            f"AdvancedUpscaling initialized: "
            f"cache={enable_cache}, workers={max_workers}, "
            f"validate={validate_images}, enhance={enhance_images}, "
            f"auto_select={auto_select_method}"
        )
    
    def _select_best_method(self, image: Image.Image, scale_factor: float) -> str:
        """Select best method based on image characteristics."""
        return MethodSelector.select_best_method(
            image, scale_factor, auto_select=self.auto_select_method, stats=self.stats
        )
    
    @staticmethod
    def calculate_quality_metrics(image: Image.Image) -> QualityMetrics:
        """Calculate quality metrics for an image."""
        return QualityCalculator.calculate_quality_metrics(image)
    
    def get_optimal_resolution(
        self,
        original_size: Tuple[int, int],
        scale_factor: float,
        max_dimension: Optional[int] = None
    ) -> Tuple[int, int]:
        """Calculate optimal resolution for upscaling."""
        width, height = original_size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        if max_dimension:
            if new_width > max_dimension or new_height > max_dimension:
                if new_width > new_height:
                    ratio = max_dimension / new_width
                else:
                    ratio = max_dimension / new_height
                new_width = int(new_width * ratio)
                new_height = int(new_height * ratio)
        
        return (new_width, new_height)
    
    @staticmethod
    def upscale_with_post_processing(
        image: Image.Image,
        scale_factor: float,
        method: str = "lanczos",
        apply_denoising: bool = True,
        apply_sharpening: bool = True,
        apply_anti_aliasing: bool = False
    ) -> Image.Image:
        """Upscale with comprehensive post-processing."""
        # Upscale
        if method == "lanczos":
            result = UpscalingAlgorithmsStatic.upscale_lanczos(image, scale_factor)
        elif method == "bicubic":
            result = UpscalingAlgorithmsStatic.upscale_bicubic_enhanced(image, scale_factor)
        elif method == "opencv":
            result = UpscalingAlgorithmsStatic.upscale_opencv_edsr(image, scale_factor)
        elif method == "multi_scale":
            result = UpscalingAlgorithmsStatic.multi_scale_upscale(image, scale_factor)
        else:
            result = UpscalingAlgorithmsStatic.upscale_lanczos(image, scale_factor)
        
        # Post-processing
        if apply_denoising:
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral")
        if apply_sharpening:
            result = PostprocessingMethods.enhance_edges(result, strength=1.1)
        if apply_anti_aliasing:
            result = PostprocessingMethods.apply_anti_aliasing(result, strength=0.3)
        
        return result
    
    # Delegate static methods to algorithms
    @staticmethod
    def upscale_lanczos(image: Image.Image, scale_factor: float, taps: int = 3) -> Image.Image:
        """Upscale using Lanczos."""
        return UpscalingAlgorithmsStatic.upscale_lanczos(image, scale_factor, taps)
    
    @staticmethod
    def upscale_bicubic_enhanced(image: Image.Image, scale_factor: float) -> Image.Image:
        """Enhanced bicubic upscaling."""
        return UpscalingAlgorithmsStatic.upscale_bicubic_enhanced(image, scale_factor)
    
    @staticmethod
    def upscale_opencv_edsr(image: Image.Image, scale_factor: float) -> Image.Image:
        """OpenCV EDSR upscaling."""
        return UpscalingAlgorithmsStatic.upscale_opencv_edsr(image, scale_factor)
    
    @staticmethod
    def multi_scale_upscale(image: Image.Image, scale_factor: float, passes: int = 2) -> Image.Image:
        """Multi-scale upscaling."""
        return UpscalingAlgorithmsStatic.multi_scale_upscale(image, scale_factor, passes)
    
    @staticmethod
    def upscale_adaptive(image: Image.Image, scale_factor: float) -> Image.Image:
        """Adaptive upscaling."""
        return UpscalingAlgorithmsStatic.upscale_adaptive(image, scale_factor)


# Export for backward compatibility
__all__ = [
    "AdvancedUpscalingRefactored",
    "REALESRGAN_AVAILABLE",
    "RealESRGANUpscaler",
]


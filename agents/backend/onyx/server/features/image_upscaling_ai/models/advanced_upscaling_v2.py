"""
Advanced Upscaling - Version 2 (Refactored with Mixins)
========================================================

This is a refactored version of advanced_upscaling.py that uses mixins
for better modularity and maintainability.

The original advanced_upscaling.py is kept for backward compatibility.
"""

import logging
from typing import Tuple, Optional, Dict, Any, List, Callable, Union
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Import all mixins
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
    SpecializedMixin,
    ExportMixin,
    ConfigurationMixin,
    BenchmarkMixin,
    ValidationMixin,
    MonitoringMixin,
    LearningMixin,
    IntegrationMixin,
    SecurityMixin,
    CompressionMixin,
    PerformanceMixin,
    WorkflowMixin,
    ExperimentationMixin,
    StreamingMixin,
    BackupMixin,
    VersioningMixin,
    HistoryMixin,
    NotificationMixin,
    CollaborationMixin,
    SyncMixin,
    TrendsMixin,
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


class AdvancedUpscalingV2(
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
    SpecializedMixin,
    ExportMixin,
    ConfigurationMixin,
    BenchmarkMixin,
    ValidationMixin,
    MonitoringMixin,
    LearningMixin,
    IntegrationMixin,
    SecurityMixin,
    CompressionMixin,
    PerformanceMixin,
    WorkflowMixin,
    ExperimentationMixin,
    StreamingMixin,
    BackupMixin,
    VersioningMixin,
    HistoryMixin,
    NotificationMixin,
    CollaborationMixin,
    SyncMixin,
    TrendsMixin
):
    """
    Advanced upscaling techniques - Version 2 (Refactored with Mixins).
    
    This class combines all mixins to provide a complete upscaling solution.
    
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
    - Optimization
    - Quality assurance
    - Utilities
    
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
    - Specialized upscaling (face, text, artwork, photo, anime)
    - Export functionality
    - Configuration and preset management
    - Benchmarking and performance testing
    - Advanced validation and verification
    - Monitoring and logging
    - Machine learning and adaptive learning
    - API and external service integration
    - Security and safety features
    - Image compression and optimization
    - Performance profiling and optimization
    - Workflow orchestration and automation
    - A/B testing and experimentation
    - Streaming and real-time processing
    - Backup and restore functionality
    - Versioning and history tracking
    - Operation history and audit trail
    - Notification and alert system
    - Collaboration and sharing
    - Synchronization
    - Trends analysis and insights
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
        Initialize Advanced Upscaling V2.
        
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
            f"AdvancedUpscalingV2 initialized: "
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


# Export for backward compatibility
__all__ = [
    "AdvancedUpscalingV2",
    "REALESRGAN_AVAILABLE",
    "RealESRGANUpscaler",
]

# Alias for easier migration
AdvancedUpscaling = AdvancedUpscalingV2


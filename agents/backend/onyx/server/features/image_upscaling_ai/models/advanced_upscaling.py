"""
Advanced Upscaling Techniques
==============================

Advanced image upscaling algorithms and techniques with caching, async processing,
quality metrics, and performance optimizations.

This module maintains backward compatibility by using the refactored V2 version
with mixins. All functionality is available through the mixin-based architecture.

The refactored version (AdvancedUpscalingV2) uses 26 mixins providing 130+ methods
for complete upscaling functionality including:
- Core upscaling
- Image enhancement
- ML/AI methods
- Analysis and reporting
- Pipeline management
- Batch processing
- Caching
- Optimization
- Quality assurance
- Specialized upscaling
- Export functionality
- Configuration management
- Benchmarking
- Validation
- Monitoring
- Learning
- Integration
- Security
- Compression
- Performance profiling
- Workflow orchestration
- A/B testing
- Streaming
- Backup and restore
"""

import logging
from typing import Tuple, Optional, Dict, Any, List, Callable, Union
from pathlib import Path
from PIL import Image

# Import the refactored V2 version (with mixins)
from .advanced_upscaling_v2 import AdvancedUpscalingV2

# Import helpers for backward compatibility
from .helpers import (
    UpscalingMetrics,
    QualityMetrics,
)

logger = logging.getLogger(__name__)

# Try to import Real-ESRGAN
try:
    from .realesrgan_integration import RealESRGANUpscaler, REALESRGAN_AVAILABLE
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANUpscaler = None

if not REALESRGAN_AVAILABLE:
    logger.warning("Real-ESRGAN not available. Install with: pip install realesrgan basicsr")


class AdvancedUpscaling(AdvancedUpscalingV2):
    """
    Advanced upscaling techniques - backward compatible wrapper.
    
    This class extends AdvancedUpscalingV2 (which uses 26 mixins) to provide
    full backward compatibility. All methods from the mixins are directly available.
    
    The mixin-based architecture provides:
    - 26 specialized mixins
    - 130+ methods
    - Complete upscaling functionality
    - Modular and maintainable code
    
    Features:
    - Multiple upscaling algorithms (Lanczos, Bicubic, OpenCV, Multi-scale, etc.)
    - Real-ESRGAN integration
    - Face, text, artwork, photo, and anime specialized upscaling
    - Batch processing with progress tracking
    - Intelligent caching system
    - Quality metrics and validation
    - Performance optimization
    - Workflow orchestration
    - A/B testing and experimentation
    - Streaming and real-time processing
    
    Usage:
        from image_upscaling_ai.models import AdvancedUpscaling
        
        # Initialize with default settings
        upscaler = AdvancedUpscaling()
        
        # Or with custom configuration
        upscaler = AdvancedUpscaling(
            enable_cache=True,
            cache_size=128,
            max_workers=8,
            auto_select_method=True
        )
        
        # Basic upscaling
        result = upscaler.upscale("image.jpg", 2.0)
        
        # Specialized upscaling
        result = upscaler.upscale_face("portrait.jpg", 2.0)
        result = upscaler.upscale_anime("anime.jpg", 2.0)
        result = upscaler.upscale_photo("photo.jpg", 2.0)
        
        # With quality metrics
        result, metrics = upscaler.upscale("image.jpg", 2.0, return_metrics=True)
        
        # Batch processing
        results = upscaler.batch_upscale(["img1.jpg", "img2.jpg"], 2.0)
        
        # Analysis and recommendations
        analysis = upscaler.analyze_image_characteristics("image.jpg")
        recommendations = upscaler.get_processing_recommendations(image, 2.0)
        
        # Benchmarking
        benchmark = upscaler.benchmark_methods("image.jpg", 2.0)
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
            enable_cache: Enable result caching for faster repeated operations
            cache_size: Maximum number of cached results (default: 64)
            cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
            max_workers: Maximum worker threads for parallel processing (default: 4)
            validate_images: Validate images before processing (default: True)
            enhance_images: Automatically enhance images before upscaling (default: False)
            auto_select_method: Automatically select best method based on image (default: False)
            max_retries: Maximum retry attempts for failed operations (default: 3)
        
        Example:
            >>> upscaler = AdvancedUpscaling(
            ...     enable_cache=True,
            ...     cache_size=128,
            ...     max_workers=8,
            ...     auto_select_method=True
            ... )
        """
        super().__init__(
            enable_cache=enable_cache,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            max_workers=max_workers,
            validate_images=validate_images,
            enhance_images=enhance_images,
            auto_select_method=auto_select_method,
            max_retries=max_retries,
        )
        logger.info(
            f"AdvancedUpscaling initialized (using V2 with mixins): "
            f"cache={enable_cache}, workers={max_workers}, "
            f"validate={validate_images}, enhance={enhance_images}, "
            f"auto_select={auto_select_method}"
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available capabilities and features.
        
        Returns:
            Dictionary with information about available methods, algorithms,
            and features.
        
        Example:
            >>> upscaler = AdvancedUpscaling()
            >>> caps = upscaler.get_capabilities()
            >>> print(caps['algorithms'])
        """
        return {
            "version": "2.0",
            "architecture": "mixin-based",
            "mixins_count": 26,
            "methods_count": "130+",
            "algorithms": [
                "lanczos",
                "bicubic",
                "opencv",
                "multi_scale",
                "adaptive",
                "esrgan_like",
                "waifu2x_like",
                "real_esrgan_like",
            ],
            "specialized_types": [
                "face",
                "text",
                "artwork",
                "photo",
                "anime",
            ],
            "features": [
                "batch_processing",
                "caching",
                "quality_metrics",
                "performance_optimization",
                "workflow_orchestration",
                "a_b_testing",
                "streaming",
                "backup_restore",
                "real_esrgan_integration",
            ],
            "realesrgan_available": REALESRGAN_AVAILABLE,
        }
    
    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific upscaling method.
        
        Args:
            method_name: Name of the method to get info for
        
        Returns:
            Dictionary with method information or None if method not found
        
        Example:
            >>> upscaler = AdvancedUpscaling()
            >>> info = upscaler.get_method_info("lanczos")
        """
        methods_info = {
            "lanczos": {
                "name": "Lanczos",
                "description": "High-quality resampling algorithm, good for general use",
                "speed": "fast",
                "quality": "high",
                "best_for": ["general", "photos", "artwork"],
            },
            "bicubic": {
                "name": "Bicubic Enhanced",
                "description": "Enhanced bicubic interpolation with artifact reduction",
                "speed": "fast",
                "quality": "medium-high",
                "best_for": ["general", "photos"],
            },
            "opencv": {
                "name": "OpenCV EDSR-like",
                "description": "OpenCV-based upscaling with EDSR-like quality",
                "speed": "medium",
                "quality": "high",
                "best_for": ["photos", "artwork"],
            },
            "multi_scale": {
                "name": "Multi-scale",
                "description": "Progressive multi-scale upscaling for large scale factors",
                "speed": "slow",
                "quality": "very_high",
                "best_for": ["large_scale_factors", "high_quality"],
            },
            "adaptive": {
                "name": "Adaptive",
                "description": "Adaptive method selection based on image characteristics",
                "speed": "medium",
                "quality": "high",
                "best_for": ["auto_selection", "mixed_content"],
            },
            "esrgan_like": {
                "name": "ESRGAN-like",
                "description": "ESRGAN-inspired upscaling algorithm",
                "speed": "slow",
                "quality": "very_high",
                "best_for": ["artwork", "anime", "high_quality"],
            },
            "waifu2x_like": {
                "name": "Waifu2x-like",
                "description": "Waifu2x-inspired algorithm optimized for anime/artwork",
                "speed": "slow",
                "quality": "very_high",
                "best_for": ["anime", "artwork", "illustrations"],
            },
            "real_esrgan_like": {
                "name": "Real-ESRGAN-like",
                "description": "Real-ESRGAN-inspired algorithm for general images",
                "speed": "slow",
                "quality": "very_high",
                "best_for": ["general", "photos", "artwork"],
            },
        }
        
        return methods_info.get(method_name.lower())
    
    def list_available_methods(self) -> List[str]:
        """
        List all available upscaling methods.
        
        Returns:
            List of method names
        
        Example:
            >>> upscaler = AdvancedUpscaling()
            >>> methods = upscaler.list_available_methods()
        """
        return [
            "lanczos",
            "bicubic",
            "opencv",
            "multi_scale",
            "adaptive",
            "esrgan_like",
            "waifu2x_like",
            "real_esrgan_like",
        ]
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of processing statistics.
        
        Returns:
            Dictionary with statistics summary
        
        Example:
            >>> upscaler = AdvancedUpscaling()
            >>> # ... process some images ...
            >>> stats = upscaler.get_statistics_summary()
        """
        if hasattr(self, 'stats') and self.stats:
            return {
                "total_upscales": self.stats.get("total_upscales", 0),
                "successful_upscales": self.stats.get("successful_upscales", 0),
                "failed_upscales": self.stats.get("failed_upscales", 0),
                "cache_hits": self.stats.get("cache_hits", 0),
                "cache_misses": self.stats.get("cache_misses", 0),
                "total_time": self.stats.get("total_time", 0.0),
                "average_time": (
                    self.stats.get("total_time", 0.0) / max(self.stats.get("total_upscales", 1), 1)
                ),
            }
        return {
            "total_upscales": 0,
            "successful_upscales": 0,
            "failed_upscales": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0,
            "average_time": 0.0,
        }
    
    def reset_statistics(self) -> None:
        """
        Reset all processing statistics.
        
        Example:
            >>> upscaler = AdvancedUpscaling()
            >>> # ... process images ...
            >>> upscaler.reset_statistics()
        """
        if hasattr(self, 'stats') and self.stats:
            self.stats = {
                "total_upscales": 0,
                "successful_upscales": 0,
                "failed_upscales": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_time": 0.0,
            }
            logger.info("Statistics reset")


# Export all for backward compatibility
__all__ = [
    "AdvancedUpscaling",
    "UpscalingMetrics",
    "QualityMetrics",
    "REALESRGAN_AVAILABLE",
    "RealESRGANUpscaler",
]

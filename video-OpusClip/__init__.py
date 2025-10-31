"""
Video Processing Module

Ultra-fast, batch-optimized video processing with modular architecture and advanced parallel processing:
- Batch operations and utilities
- Validation and metrics
- Video models and processors
- Viral content generation
- Advanced parallel processing with multiple backends
- Image and caption processing

Usage:
    from agents.backend.onyx.server.features.video import (
        VideoClipRequest, VideoClip, VideoClipResponse,
        ViralClipVariant, ViralVideoBatchResponse,
        VideoClipProcessor, ViralVideoProcessor,
        parallel_map, HybridParallelProcessor
    )
"""

# =============================================================================
# CORE MODELS
# =============================================================================

from .models.video_models import (
    VideoClipRequest,
    VideoClip, 
    VideoClipResponse
)

from .models.viral_models import (
    ViralClipVariant,
    CaptionOutput,
    ViralVideoBatchResponse
)

# =============================================================================
# PROCESSORS
# =============================================================================

from .processors.video_processor import (
    VideoClipProcessor,
    VideoEncodingProcessor,
    VideoValidationProcessor,
    create_video_processor,
    create_high_performance_processor
)

from .processors.viral_processor import (
    ViralVideoProcessor,
    ViralVariantProcessor,
    ViralAnalyticsProcessor,
    create_viral_processor,
    create_high_performance_viral_processor
)

# =============================================================================
# UTILITIES
# =============================================================================

from .utils.batch_utils import (
    BatchMetrics,
    _optimized_batch_timeit,
    _vectorized_batch_operation,
    _parallel_batch_operation,
    OptimizedBatchMixin
)

from .utils.validation import (
    _validate_youtube_url,
    _validate_language,
    _validate_video_clip_times,
    _validate_viral_score,
    _validate_caption
)

from .utils.parallel_utils import (
    HybridParallelProcessor,
    VideoParallelProcessor,
    parallel_map,
    async_batch_process,
    joblib_parallel_process,
    ray_parallel_process,
    dask_parallel_process,
    numba_parallel_process,
    ParallelConfig,
    BackendType,
    setup_async_loop,
    get_optimal_chunk_size,
    estimate_processing_time
)

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

from .utils.constants import (
    PANDAS_AVAILABLE,
    SENTRY_AVAILABLE,
    SLOW_OPERATION_THRESHOLD,
    DEFAULT_MAX_WORKERS,
    CACHE_SIZE_URLS,
    CACHE_SIZE_LANGUAGES
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__version__ = "2.0.0"
__author__ = "Onyx Team"

__all__ = [
    # Core video models
    'VideoClipRequest',
    'VideoClip',
    'VideoClipResponse',
    
    # Viral models
    'ViralClipVariant',
    'CaptionOutput', 
    'ViralVideoBatchResponse',
    
    # Processors
    'VideoClipProcessor',
    'VideoEncodingProcessor',
    'VideoValidationProcessor',
    'ViralVideoProcessor',
    'ViralVariantProcessor',
    'ViralAnalyticsProcessor',
    
    # Factory functions
    'create_video_processor',
    'create_high_performance_processor',
    'create_viral_processor',
    'create_high_performance_viral_processor',
    
    # Batch utilities
    'BatchMetrics',
    '_optimized_batch_timeit',
    '_vectorized_batch_operation',
    '_parallel_batch_operation',
    'OptimizedBatchMixin',
    
    # Validation
    '_validate_youtube_url',
    '_validate_language',
    '_validate_video_clip_times',
    '_validate_viral_score',
    '_validate_caption',
    
    # Parallel processing
    'HybridParallelProcessor',
    'VideoParallelProcessor',
    'parallel_map',
    'async_batch_process',
    'joblib_parallel_process',
    'ray_parallel_process',
    'dask_parallel_process',
    'numba_parallel_process',
    'ParallelConfig',
    'BackendType',
    'setup_async_loop',
    'get_optimal_chunk_size',
    'estimate_processing_time',
    
    # Constants
    'PANDAS_AVAILABLE',
    'SENTRY_AVAILABLE',
    'SLOW_OPERATION_THRESHOLD',
    'DEFAULT_MAX_WORKERS',
    'CACHE_SIZE_URLS',
    'CACHE_SIZE_LANGUAGES',
] 
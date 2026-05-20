"""
Utility functions
Performance optimization and utility functions
"""

# Existing utilities
from .advanced_validator import AdvancedImageValidator
from .distributed_cache import DistributedCache
from .advanced_logging import AdvancedLogger, LogLevel
from .intelligent_cache import IntelligentCache
from .endpoint_rate_limiter import EndpointRateLimiter

# Performance optimization
try:
    from .optimization import (
        compile_model,
        optimize_for_inference,
        fuse_model,
        quantize_model,
        enable_cudnn_benchmark,
        enable_tf32,
        optimize_data_loading,
        FastInferenceEngine,
        create_optimized_trainer,
        optimize_memory,
        set_optimal_threads,
        ModelCache
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Async inference
try:
    from .async_inference import (
        AsyncInferenceEngine,
        BatchInferenceEngine
    )
    ASYNC_INFERENCE_AVAILABLE = True
except ImportError:
    ASYNC_INFERENCE_AVAILABLE = False

# Profiling
try:
    from .profiling import (
        profile_region,
        time_function,
        PerformanceMonitor,
        profile_model,
        optimize_data_loader,
        check_gpu_utilization,
        clear_gpu_cache
    )
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Advanced optimization
try:
    from .advanced_optimization import (
        enable_gradient_checkpointing,
        enable_flash_attention,
        enable_memory_efficient_attention,
        MemoryEfficientModel,
        optimize_tensor_operations,
        SmartBatchProcessor,
        LazyModelLoader,
        optimize_preprocessing_pipeline,
        TensorPool,
        enable_all_optimizations,
        OptimizedDataLoader
    )
    ADVANCED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZATION_AVAILABLE = False

# Model pruning
try:
    from .model_pruning import (
        prune_model,
        get_model_size,
        KnowledgeDistillation,
        create_quantized_model,
        compare_models
    )
    MODEL_PRUNING_AVAILABLE = True
except ImportError:
    MODEL_PRUNING_AVAILABLE = False

__all__ = [
    # Existing
    "AdvancedImageValidator",
    "DistributedCache",
    "AdvancedLogger",
    "LogLevel",
    "IntelligentCache",
    "EndpointRateLimiter",
]

# Add optimization exports if available
if OPTIMIZATION_AVAILABLE:
    __all__.extend([
        'compile_model',
        'optimize_for_inference',
        'fuse_model',
        'quantize_model',
        'enable_cudnn_benchmark',
        'enable_tf32',
        'optimize_data_loading',
        'FastInferenceEngine',
        'create_optimized_trainer',
        'optimize_memory',
        'set_optimal_threads',
        'ModelCache',
    ])

# Add async inference exports if available
if ASYNC_INFERENCE_AVAILABLE:
    __all__.extend([
        'AsyncInferenceEngine',
        'BatchInferenceEngine',
    ])

# Add profiling exports if available
if PROFILING_AVAILABLE:
    __all__.extend([
        'profile_region',
        'time_function',
        'PerformanceMonitor',
        'profile_model',
        'optimize_data_loader',
        'check_gpu_utilization',
        'clear_gpu_cache'
    ])

# Add advanced optimization exports if available
if ADVANCED_OPTIMIZATION_AVAILABLE:
    __all__.extend([
        'enable_gradient_checkpointing',
        'enable_flash_attention',
        'enable_memory_efficient_attention',
        'MemoryEfficientModel',
        'optimize_tensor_operations',
        'SmartBatchProcessor',
        'LazyModelLoader',
        'optimize_preprocessing_pipeline',
        'TensorPool',
        'enable_all_optimizations',
        'OptimizedDataLoader'
    ])

# Add model pruning exports if available
if MODEL_PRUNING_AVAILABLE:
    __all__.extend([
        'prune_model',
        'get_model_size',
        'KnowledgeDistillation',
        'create_quantized_model',
        'compare_models'
    ])

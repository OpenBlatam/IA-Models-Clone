"""
Utils module for Lovable Community

Enhanced utilities for performance, validation, retry, batch processing, and more.
Maintains backward compatibility with existing utilities.
"""

# New enhanced utilities
from .performance import (
    timer,
    measure_time,
    PerformanceMonitor,
    get_performance_monitor
)

from .retry import (
    retry,
    RetryableOperation
)

from .validation import (
    validate_not_none,
    validate_not_empty,
    validate_length,
    validate_range,
    validate_one_of,
    validate_pattern,
    validate_datetime_range,
    validate_custom
)

from .batch import (
    batch_process,
    chunk_list,
    bulk_create,
    bulk_update
)

from .serialization import (
    fast_json_dumps,
    fast_json_loads,
    serialize_model_fast,
    serialize_models_batch
)

from .model_loading import (
    load_model_optimized,
    load_model_quantized
)

from .inference import (
    inference_mode,
    generate_text_optimized,
    batch_inference,
    compile_model
)

from .inference_optimized import (
    FastInferenceEngine,
    ONNXRuntimeEngine,
    InferenceCache,
    fast_generate,
    optimize_model_for_inference,
    MemoryPool
)

from .data_loading_optimized import (
    OptimizedDataLoader,
    FastCollate,
    CachedDataset,
    get_optimal_num_workers,
    PrefetchDataset
)

from .fused_operations import (
    FusedLayerNorm,
    FusedLinearGELU,
    FusedAttention,
    fuse_conv_bn,
    optimize_model_fused
)

from .vectorization import (
    vectorized_embedding_lookup,
    vectorized_batch_norm,
    vectorized_softmax,
    vectorized_matmul_attention,
    VectorizedOperations
)

from .cuda_optimizations import (
    enable_tensor_cores,
    optimize_cuda_settings,
    get_optimal_cuda_device,
    pin_memory_tensor,
    async_copy_to_device,
    CUDAMemoryManager,
    compile_with_triton
)

from .distributed_training import (
    init_distributed,
    wrap_model_ddp,
    all_reduce_mean,
    gather_tensors,
    is_main_process,
    cleanup_distributed
)

from .monitoring import (
    PerformanceMonitor,
    TrainingProfiler,
    profile_operation,
    log_gpu_memory
)

from .model_serving import (
    ModelServer,
    export_to_onnx,
    export_to_torchscript
)

# Existing utilities (backward compatibility)
try:
    from .performance import (
        measure_execution_time,
        time_function,
        retry_on_failure,
        batch_process as old_batch_process
    )
except ImportError:
    # If old functions don't exist, that's okay
    pass

try:
    from .security import (
        sanitize_html,
        sanitize_sql_input,
        validate_email,
        validate_url,
        sanitize_filename,
        rate_limit_key,
        generate_csrf_token,
        validate_csrf_token
    )
except ImportError:
    pass

try:
    from .response_helpers import (
        add_cache_headers,
        add_cors_headers,
        add_security_headers,
        create_error_response,
        create_success_response,
        add_pagination_headers
    )
except ImportError:
    pass

try:
    from .logging_config import (
        setup_logging,
        StructuredFormatter,
        get_logger,
        PerformanceLogger,
        performance_logger
    )
except ImportError:
    pass

__all__ = [
    # Enhanced Performance
    "timer",
    "measure_time",
    "PerformanceMonitor",
    "get_performance_monitor",
    # Enhanced Retry
    "retry",
    "RetryableOperation",
    # Enhanced Validation
    "validate_not_none",
    "validate_not_empty",
    "validate_length",
    "validate_range",
    "validate_one_of",
    "validate_pattern",
    "validate_datetime_range",
    "validate_custom",
    # Batch Processing
    "batch_process",
    "chunk_list",
    "bulk_create",
    "bulk_update",
    # Fast Serialization
    "fast_json_dumps",
    "fast_json_loads",
    "serialize_model_fast",
    "serialize_models_batch",
    # Model Loading
    "load_model_optimized",
    "load_model_quantized",
    # Inference
    "inference_mode",
    "generate_text_optimized",
    "batch_inference",
    "compile_model",
    # Ultra-Fast Inference
    "FastInferenceEngine",
    "ONNXRuntimeEngine",
    "InferenceCache",
    "fast_generate",
    "optimize_model_for_inference",
    "MemoryPool",
    # Optimized Data Loading
    "OptimizedDataLoader",
    "FastCollate",
    "CachedDataset",
    "get_optimal_num_workers",
    "PrefetchDataset",
    # Fused Operations
    "FusedLayerNorm",
    "FusedLinearGELU",
    "FusedAttention",
    "fuse_conv_bn",
    "optimize_model_fused",
    # Vectorization
    "vectorized_embedding_lookup",
    "vectorized_batch_norm",
    "vectorized_softmax",
    "vectorized_matmul_attention",
    "VectorizedOperations",
    # CUDA Optimizations
    "enable_tensor_cores",
    "optimize_cuda_settings",
    "get_optimal_cuda_device",
    "pin_memory_tensor",
    "async_copy_to_device",
    "CUDAMemoryManager",
    "compile_with_triton",
    # Distributed Training
    "init_distributed",
    "wrap_model_ddp",
    "all_reduce_mean",
    "gather_tensors",
    "is_main_process",
    "cleanup_distributed",
    # Monitoring
    "PerformanceMonitor",
    "TrainingProfiler",
    "profile_operation",
    "log_gpu_memory",
    # Model Serving
    "ModelServer",
    "export_to_onnx",
    "export_to_torchscript",
]

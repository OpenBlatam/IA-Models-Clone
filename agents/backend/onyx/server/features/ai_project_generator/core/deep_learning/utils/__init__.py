"""
Utilities Module - Deep Learning Utilities
===========================================

Common utilities for deep learning workflows:
- Device management
- Experiment tracking
- Logging utilities
- Debugging tools
"""

from typing import Optional
import torch

from .device_utils import get_device, set_seed, enable_anomaly_detection
from .experiment_tracking import ExperimentTracker

# Try to import memory optimization
try:
    from .memory_optimization import (
        clear_cache,
        enable_gradient_checkpointing,
        optimize_model_for_inference,
        get_memory_stats,
        print_memory_stats,
        MemoryMonitor
    )
    MEMORY_OPT_AVAILABLE = True
except ImportError:
    MEMORY_OPT_AVAILABLE = False
    clear_cache = None
    enable_gradient_checkpointing = None
    optimize_model_for_inference = None
    get_memory_stats = None
    print_memory_stats = None
    MemoryMonitor = None

# Try to import error handling
try:
    from .error_handling import (
        ErrorHandler,
        retry_on_error,
        handle_cuda_errors,
        GracefulDegradation
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    ErrorHandler = None
    retry_on_error = None
    handle_cuda_errors = None
    GracefulDegradation = None

# Try to import additional utilities
try:
    from .profiling import profile_operation, profile_model, check_for_bottlenecks
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    profile_operation = None
    profile_model = None
    check_for_bottlenecks = None

try:
    from .validation import (
        validate_model_inputs,
        check_gradients,
        validate_data_loader,
        validate_config
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    validate_model_inputs = None
    check_gradients = None
    validate_data_loader = None
    validate_config = None

__all__ = [
    "get_device",
    "set_seed",
    "enable_anomaly_detection",
    "ExperimentTracker",
]

if PROFILING_AVAILABLE:
    __all__.extend(["profile_operation", "profile_model", "check_for_bottlenecks"])

if VALIDATION_AVAILABLE:
    __all__.extend([
        "validate_model_inputs",
        "check_gradients",
        "validate_data_loader",
        "validate_config"
    ])

if MEMORY_OPT_AVAILABLE:
    __all__.extend([
        "clear_cache",
        "enable_gradient_checkpointing",
        "optimize_model_for_inference",
        "get_memory_stats",
        "print_memory_stats",
        "MemoryMonitor"
    ])

if ERROR_HANDLING_AVAILABLE:
    __all__.extend([
        "ErrorHandler",
        "retry_on_error",
        "handle_cuda_errors",
        "GracefulDegradation"
    ])

# Try to import model analysis
try:
    from .model_analysis import (
        analyze_model_complexity,
        analyze_gradient_flow,
        get_layer_output_shapes,
        check_model_health
    )
    MODEL_ANALYSIS_AVAILABLE = True
except ImportError:
    MODEL_ANALYSIS_AVAILABLE = False
    analyze_model_complexity = None
    analyze_gradient_flow = None
    get_layer_output_shapes = None
    check_model_health = None

# Try to import checkpoint utils
try:
    from .checkpoint_utils import CheckpointManager
    CHECKPOINT_UTILS_AVAILABLE = True
except ImportError:
    CHECKPOINT_UTILS_AVAILABLE = False
    CheckpointManager = None

if MODEL_ANALYSIS_AVAILABLE:
    __all__.extend([
        "analyze_model_complexity",
        "analyze_gradient_flow",
        "get_layer_output_shapes",
        "check_model_health"
    ])

if CHECKPOINT_UTILS_AVAILABLE:
    __all__.append("CheckpointManager")

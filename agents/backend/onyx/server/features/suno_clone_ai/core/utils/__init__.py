"""
Utilities Module

Provides:
- Model utilities
- Device utilities
- Batch utilities
- File utilities
- Mixed precision utilities
"""

from .model_utils import (
    initialize_weights,
    clip_gradients,
    check_for_nan_inf,
    get_device,
    setup_gpu_optimizations,
    compile_model,
    enable_gradient_checkpointing,
    get_model_size_mb,
    clear_gpu_cache,
    set_seed
)

from .device_utils import (
    DeviceManager,
    get_best_device,
    get_device_info,
    clear_gpu_cache as clear_device_cache
)

from .batch_utils import (
    BatchProcessor,
    create_batch_processor,
    process_in_batches
)

from .file_utils import (
    FileManager,
    ensure_dir,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_pickle,
    load_pickle
)

from .mixed_precision import (
    MixedPrecisionManager,
    create_mixed_precision_manager
)

__all__ = [
    # Model utilities
    "initialize_weights",
    "clip_gradients",
    "check_for_nan_inf",
    "get_device",
    "setup_gpu_optimizations",
    "compile_model",
    "enable_gradient_checkpointing",
    "get_model_size_mb",
    "clear_gpu_cache",
    "set_seed",
    # Device utilities
    "DeviceManager",
    "get_best_device",
    "get_device_info",
    "clear_device_cache",
    # Batch utilities
    "BatchProcessor",
    "create_batch_processor",
    "process_in_batches",
    # File utilities
    "FileManager",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "save_pickle",
    "load_pickle",
    # Mixed precision
    "MixedPrecisionManager",
    "create_mixed_precision_manager"
]

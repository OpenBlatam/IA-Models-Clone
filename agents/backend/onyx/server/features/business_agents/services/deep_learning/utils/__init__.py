"""Utility modules for deep learning service."""

from .distributed import setup_ddp, cleanup_ddp, is_distributed
from .profiling import profile_training, profile_inference
from .helpers import (
    set_seed, get_device, count_parameters,
    format_size, get_model_size, save_model_summary
)

# Optional visualization imports
try:
    from .visualization import TrainingVisualizer, visualize_predictions
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    TrainingVisualizer = None
    visualize_predictions = None

# Optional optimization imports
try:
    from .optimization import (
        ModelOptimizer, MemoryOptimizer, InferenceOptimizer,
        DataLoaderOptimizer, optimize_model_for_production
    )
    from .batch_optimization import (
        BatchProcessor, InferenceCache, OptimizedInference
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    ModelOptimizer = None
    MemoryOptimizer = None
    InferenceOptimizer = None
    DataLoaderOptimizer = None
    optimize_model_for_production = None
    BatchProcessor = None
    InferenceCache = None
    OptimizedInference = None

__all__ = [
    "setup_ddp",
    "cleanup_ddp",
    "is_distributed",
    "profile_training",
    "profile_inference",
    "set_seed",
    "get_device",
    "count_parameters",
    "format_size",
    "get_model_size",
    "save_model_summary",
]

if OPTIMIZATION_AVAILABLE:
    __all__.extend([
        "ModelOptimizer",
        "MemoryOptimizer",
        "InferenceOptimizer",
        "DataLoaderOptimizer",
        "optimize_model_for_production",
        "BatchProcessor",
        "InferenceCache",
        "OptimizedInference",
    ])

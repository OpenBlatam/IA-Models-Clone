"""
Utilities Module

General utilities organized into sub-modules:
- debugging: NaN/Inf detection, gradient checking, anomaly detection
- profiling: Memory and performance profiling
- visualization: Training curves, model visualization
- gpu: Multi-GPU and distributed training
"""

from .debugging import (
    NaNInfDetector,
    GradientChecker,
    detect_anomaly,
    enable_debug_mode,
    disable_debug_mode
)

from .profiling import (
    MemoryProfiler,
    PerformanceProfiler
)

from .visualization import (
    TrainingVisualizer,
    ModelVisualizer,
    MetricsVisualizer
)

from .gpu import (
    MultiGPUTrainer,
    init_distributed,
    cleanup_distributed,
    gradient_checkpointing
)

__all__ = [
    # Debugging
    "NaNInfDetector",
    "GradientChecker",
    "detect_anomaly",
    "enable_debug_mode",
    "disable_debug_mode",
    # Profiling
    "MemoryProfiler",
    "PerformanceProfiler",
    # Visualization
    "TrainingVisualizer",
    "ModelVisualizer",
    "MetricsVisualizer",
    # GPU
    "MultiGPUTrainer",
    "init_distributed",
    "cleanup_distributed",
    "gradient_checkpointing",
]


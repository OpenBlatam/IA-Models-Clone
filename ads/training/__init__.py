"""
Training package for the ads feature.

This package provides a unified training system that consolidates all scattered
training implementations into a clean, modular architecture.
"""

from .base_trainer import BaseTrainer, TrainingConfig, TrainingResult
from .pytorch_trainer import PyTorchTrainer
# Optional heavy imports guarded to allow tests to run without GPU/ML deps
try:
    from .diffusion_trainer import DiffusionTrainer  # type: ignore
except Exception:  # pragma: no cover - optional in environment without diffusers/xformers
    DiffusionTrainer = None  # type: ignore

try:
    from .multi_gpu_trainer import MultiGPUTrainer  # type: ignore
except Exception:  # pragma: no cover - optional in environment without torch distributed
    MultiGPUTrainer = None  # type: ignore
from .training_factory import TrainingFactory
from .experiment_tracker import ExperimentTracker
from .training_optimizer import TrainingOptimizer
from .logging import (
    TrainingLogger, AsyncTrainingLogger, LogLevel, TrainingPhase, MetricType,
    TrainingMetrics, ErrorLog, TrainingProgress, MemoryTracker, training_logger_context
)
from .torch_optimizer import (
    TorchOptimizationConfig,
    TorchMemoryOptimizer,
    TorchPerformanceOptimizer,
    TorchMixedPrecisionTrainer,
    TorchModelOptimizer,
    OptimizedTorchDataset,
    TorchOptimizedTrainer,
    TorchGradientAccumulator,
    TorchProfiler,
    TorchBenchmarker,
    setup_torch_optimization,
    create_optimized_dataloader,
    optimize_model_for_training,
)
from .tokenization import (
    TextPreprocessor,
    AdvancedTokenizer,
    SequenceManager,
    OptimizedAdsDataset,
    TokenizationService,
)
from .fine_tuning import (
    OptimizedFineTuningService,
)

__all__ = [
    # Base classes
    "BaseTrainer",
    "TrainingConfig", 
    "TrainingResult",
    
    # Specific trainers
    "PyTorchTrainer",
    "DiffusionTrainer",
    "MultiGPUTrainer",
    
    # Factory and utilities
    "TrainingFactory",
    "ExperimentTracker",
    "TrainingOptimizer",
    
    # Logging and monitoring
    "TrainingLogger",
    "AsyncTrainingLogger", 
    "LogLevel",
    "TrainingPhase",
    "MetricType",
    "TrainingMetrics",
    "ErrorLog",
    "TrainingProgress",
    "MemoryTracker",
    "training_logger_context",
]

__all__.extend([
    # Torch optimizer utilities
    "TorchOptimizationConfig",
    "TorchMemoryOptimizer",
    "TorchPerformanceOptimizer",
    "TorchMixedPrecisionTrainer",
    "TorchModelOptimizer",
    "OptimizedTorchDataset",
    "TorchOptimizedTrainer",
    "TorchGradientAccumulator",
    "TorchProfiler",
    "TorchBenchmarker",
    "setup_torch_optimization",
    "create_optimized_dataloader",
    "optimize_model_for_training",

    # Tokenization
    "TextPreprocessor",
    "AdvancedTokenizer",
    "SequenceManager",
    "OptimizedAdsDataset",
    "TokenizationService",

    # Fine-tuning
    "OptimizedFineTuningService",
])

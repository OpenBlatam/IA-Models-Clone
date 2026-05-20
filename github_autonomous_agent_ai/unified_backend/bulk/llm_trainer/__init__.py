"""
LLM Trainer Module - Modular Custom Trainer for Language Models
================================================================

This package provides a modular implementation of a custom trainer for
fine-tuning Large Language Models using Hugging Face Transformers.

Modules:
    - device_manager: GPU/TPU/CPU device detection and management
    - dataset_loader: JSON dataset loading and validation
    - tokenizer_utils: Tokenization utilities
    - model_loader: Model loading and configuration
    - config: Training configuration management
    - callbacks: Custom training callbacks
    - trainer: Main CustomLLMTrainer class

Usage:
    >>> from llm_trainer import CustomLLMTrainer
    >>> trainer = CustomLLMTrainer(
    ...     model_name="gpt2",
    ...     dataset_path="data.json",
    ...     output_dir="./checkpoints"
    ... )
    >>> trainer.train()

Author: BUL System
Date: 2024
"""

# Main trainer class
from .trainer import CustomLLMTrainer

# Core components
from .core import BaseLLMTrainer, TrainerFactory, ConfigBuilder

# Infrastructure components
from .device_manager import DeviceManager
from .dataset_loader import DatasetLoader
from .tokenizer_utils import TokenizerUtils
from .model_loader import ModelLoader
from .config import TrainingConfig

# Callbacks
from .callbacks import (
    TrainingProgressCallback,
    EarlyStoppingCallback,
    MemoryMonitoringCallback,
    TrainingTimeCallback,
)

# Metrics
from .metrics import compute_metrics, compute_perplexity, compute_accuracy

# Utilities
from .utils import (
    validate_dataset_path,
    validate_model_name,
    estimate_training_time,
    calculate_model_size,
    format_training_summary,
)

# Plugins
from .plugins import (
    BasePlugin,
    PluginRegistry,
    CallbackPlugin,
    MetricPlugin,
)

# Data components
from .data import (
    DatasetValidator,
    FormatValidator,
    DatasetProcessor,
    TextProcessor,
    DatasetFormatLoader,
)

# Model components
from .models import (
    ModelFactory,
    ModelConfig,
)

# Training components
from .training import (
    CheckpointManager,
    ResumeManager,
)

# Monitoring components
from .monitoring import (
    ExperimentTracker,
    PerformanceProfiler,
)

# Distributed training
from .distributed import (
    setup_distributed_training,
    is_distributed,
    get_world_size,
    get_rank,
)


__all__ = [
    # Main trainer
    "CustomLLMTrainer",
    # Core (interfaces and factories)
    "BaseLLMTrainer",
    "TrainerFactory",
    "ConfigBuilder",
    # Infrastructure
    "DeviceManager",
    "DatasetLoader",
    "TokenizerUtils",
    "ModelLoader",
    "TrainingConfig",
    # Callbacks
    "TrainingProgressCallback",
    "EarlyStoppingCallback",
    "MemoryMonitoringCallback",
    "TrainingTimeCallback",
    # Metrics
    "compute_metrics",
    "compute_perplexity",
    "compute_accuracy",
    # Utilities
    "validate_dataset_path",
    "validate_model_name",
    "estimate_training_time",
    "calculate_model_size",
    "format_training_summary",
    # Plugins
    "BasePlugin",
    "PluginRegistry",
    "CallbackPlugin",
    "MetricPlugin",
    # Data components
    "DatasetValidator",
    "FormatValidator",
    "DatasetProcessor",
    "TextProcessor",
    "DatasetFormatLoader",
    # Model components
    "ModelFactory",
    "ModelConfig",
    # Training components
    "CheckpointManager",
    "ResumeManager",
    # Monitoring components
    "ExperimentTracker",
    "PerformanceProfiler",
    # Distributed training
    "setup_distributed_training",
    "is_distributed",
    "get_world_size",
    "get_rank",
]

# Package metadata
__author__ = "BUL System"
__email__ = "bul@system.com"
__license__ = "MIT"

__version__ = "2.2.0"


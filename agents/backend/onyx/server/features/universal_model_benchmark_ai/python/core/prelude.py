"""
Prelude Module - Common imports for convenience.

This module provides a convenient way to import commonly used types and functions.
Use: from core.prelude import *
"""

# Core types
from .config import (
    DeviceType,
    QuantizationType,
    ModelConfig,
    BenchmarkConfig,
)

from .results import (
    ResultStatus,
    BenchmarkResult,
    ResultsManager,
)

from .experiments import (
    ExperimentStatus,
    ExperimentConfig,
    ExperimentManager,
)

from .model_registry import (
    ModelStatus,
    ModelRegistry,
)

# Utilities
from .utils import (
    measure_time,
    format_size,
    format_duration,
)

# Resilience
from .resilience import CircuitBreaker, retry, with_timeout

# Environment
from .environment import env_manager, Environment

# Event Bus
from .event_bus import event_bus, EventType

# Metrics
from .metrics import metrics_collector

# Performance
from .performance import performance_optimizer

__all__ = [
    # Config
    "DeviceType",
    "QuantizationType",
    "ModelConfig",
    "BenchmarkConfig",
    # Results
    "ResultStatus",
    "BenchmarkResult",
    "ResultsManager",
    # Experiments
    "ExperimentStatus",
    "ExperimentConfig",
    "ExperimentManager",
    # Model Registry
    "ModelStatus",
    "ModelRegistry",
    # Utils
    "measure_time",
    "format_size",
    "format_duration",
    # Resilience
    "CircuitBreaker",
    "retry",
    "with_timeout",
    # Environment
    "env_manager",
    "Environment",
    # Event Bus
    "event_bus",
    "EventType",
    # Metrics
    "metrics_collector",
    # Performance
    "performance_optimizer",
]


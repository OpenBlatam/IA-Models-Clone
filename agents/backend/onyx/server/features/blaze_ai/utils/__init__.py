"""
Utility modules for Blaze AI - Performance, training, monitoring, and general utilities.

This module provides a comprehensive set of utilities for:
- Performance optimization and monitoring
- Advanced training and evaluation
- Logging and metrics collection
- Caching and memory management
- Device and configuration management
"""

# =============================================================================
# Core Utilities
# =============================================================================

from .logging import get_logger, setup_logging
from .config import ConfigManager, ConfigValidator, load_config_from_file, create_config_manager
from .constants import *

# =============================================================================
# Performance and Training
# =============================================================================

from .performance_optimization import (
    PerformanceOptimizer,
    DiffusionOptimizer,
    GradioOptimizer
)

from .advanced_training import (
    TrainingConfig,
    AdvancedTrainer,
    TransformersTrainer,
    DiffusionTrainer,
    EarlyStopping
)

# =============================================================================
# Monitoring and Metrics
# =============================================================================

from .monitoring import (
    PerformanceMonitor,
    ResourceMonitor,
    TrainingMonitor,
    SystemMonitor,
    MetricPoint,
    MetricSeries
)

from .metrics import (
    AdvancedMetricsCollector,
    get_metrics_collector,
    shutdown_metrics_collector,
    track_performance,
    MetricType,
    MetricCategory,
    MetricDefinition
)

# =============================================================================
# Memory and Caching
# =============================================================================

from .cache import (
    Cache,
    LRUCache,
    TTLCache,
    FunctionCache,
    DistributedCache,
    create_cache
)

from .memory import (
    MemoryProfiler,
    MemoryOptimizer,
    MemoryManager
)

# =============================================================================
# Specialized Utilities
# =============================================================================

from .initialization import (
    ModelInitializer,
    WeightInitializer,
    TrainingUtilities
)

from .experiment import (
    ExperimentTracker,
    HyperparameterOptimizer,
    ABTestManager
)

# =============================================================================
# Export all utilities
# =============================================================================

__all__ = [
    # Core
    "get_logger",
    "setup_logging", 
    "ConfigManager",
    "ConfigValidator",
    "load_config_from_file",
    "create_config_manager",
    
    # Performance and Training
    "PerformanceOptimizer",
    "DiffusionOptimizer",
    "GradioOptimizer",
    "TrainingConfig",
    "AdvancedTrainer",
    "TransformersTrainer",
    "DiffusionTrainer",
    "EarlyStopping",
    
    # Monitoring and Metrics
    "PerformanceMonitor",
    "ResourceMonitor",
    "TrainingMonitor",
    "SystemMonitor",
    "MetricPoint",
    "MetricSeries",
    "AdvancedMetricsCollector",
    "get_metrics_collector",
    "shutdown_metrics_collector",
    "track_performance",
    "MetricType",
    "MetricCategory",
    "MetricDefinition",
    
    # Memory and Caching
    "Cache",
    "LRUCache",
    "TTLCache",
    "FunctionCache",
    "DistributedCache",
    "create_cache",
    "MemoryProfiler",
    "MemoryOptimizer",
    "MemoryManager",
    
    # Specialized
    "ModelInitializer",
    "WeightInitializer",
    "TrainingUtilities",
    "ExperimentTracker",
    "HyperparameterOptimizer",
    "ABTestManager"
]

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
    GradientAccumulator,
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
    MetricsCollector,
    TrainingMetrics,
    PerformanceMetrics,
    SystemMetrics
)

# =============================================================================
# Memory and Caching
# =============================================================================

from .cache import (
    CacheManager,
    ModelCache,
    ResultCache,
    TTLCache
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
    PipelineInitializer,
    ServiceInitializer
)

from .experiment import (
    ExperimentTracker,
    HyperparameterOptimizer,
    A/BTestManager
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
    "GradientAccumulator",
    "EarlyStopping",
    
    # Monitoring and Metrics
    "PerformanceMonitor",
    "ResourceMonitor",
    "TrainingMonitor",
    "SystemMonitor",
    "MetricPoint",
    "MetricSeries",
    "MetricsCollector",
    "TrainingMetrics",
    "PerformanceMetrics",
    "SystemMetrics",
    
    # Memory and Caching
    "CacheManager",
    "ModelCache",
    "ResultCache",
    "TTLCache",
    "MemoryProfiler",
    "MemoryOptimizer",
    "MemoryManager",
    
    # Specialized
    "ModelInitializer",
    "PipelineInitializer",
    "ServiceInitializer",
    "ExperimentTracker",
    "HyperparameterOptimizer",
    "A/BTestManager"
]

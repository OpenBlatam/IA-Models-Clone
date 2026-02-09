"""
Core system exports (validation, testing, monitoring, errors, etc.)
"""

# Validation
from .core.validation.validators import (
    InputValidator,
    ModelValidator,
    validate_input,
    validate_features,
    validate_text
)

# Testing
from .core.testing.model_tester import (
    ModelTester,
    create_model_tester
)

# Monitoring
from .core.monitoring.health_check import (
    SystemHealthMonitor,
    ModelHealthMonitor,
    create_system_monitor,
    create_model_monitor
)

# Error Handling
from .core.errors.custom_exceptions import (
    RecoveryAIError,
    ModelError,
    ModelLoadError,
    ModelInferenceError,
    ModelTrainingError,
    DataError,
    DataValidationError,
    DataProcessingError,
    ConfigurationError,
    InferenceError,
    CUDAOutOfMemoryError,
    ValidationError
)

from .core.errors.error_handler import (
    ErrorHandler,
    handle_errors,
    safe_inference
)

# Integration
from .core.integration.integrated_pipeline import (
    IntegratedPipeline,
    create_integrated_pipeline
)

# Logging
from .core.logging.structured_logger import (
    StructuredFormatter,
    ModelLogger,
    create_model_logger
)

# Benchmarking
from .core.benchmarking.benchmark import (
    ModelBenchmark,
    create_benchmark
)

# Additional Utilities
from .core.utils.performance_utils import (
    PerformanceOptimizer,
    enable_optimizations,
    profile_model
)

from .core.utils.model_initialization import (
    init_weights_xavier,
    init_weights_kaiming,
    init_weights_orthogonal,
    init_weights_normal,
    init_weights_zero,
    initialize_model
)

# Visualization
from .core.visualization.visualizer import (
    TrainingVisualizer,
    ModelVisualizer,
    create_training_visualizer,
    create_model_visualizer
)

# Security
from .core.security.model_security import (
    ModelSecurity,
    compute_model_hash,
    verify_model_integrity,
    sanitize_input
)

# Caching
from .core.caching.smart_cache import (
    SmartCache,
    create_smart_cache
)

# Advanced Metrics
from .core.metrics.advanced_metrics import (
    AdvancedMetrics,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_correlation
)

# Serialization
from .core.serialization.model_serializer import (
    ModelSerializer,
    save_model,
    load_model
)

# Export
from .core.export.model_exporter import (
    ModelExporter,
    export_to_onnx,
    export_to_torchscript
)

# Helpers
from .core.helpers.helper_functions import (
    get_device,
    count_parameters,
    freeze_layers,
    unfreeze_layers,
    set_learning_rate,
    get_learning_rate,
    format_number
)

# CI/CD
from .core.ci_cd.deployment_utils import (
    DeploymentConfig,
    HealthCheck,
    get_deployment_config,
    check_system_health,
    check_model_health
)

from .core.factories.model_factory import (
    ModelFactory,
    ModelBuilder
)

from .core.factories.trainer_factory import (
    TrainerFactory
)

from .core.config.config_loader import (
    ConfigLoader,
    get_config
)

from .core.data.data_loader_factory import (
    DataLoaderFactory
)

from .core.plugins.plugin_manager import (
    Plugin,
    PluginManager,
    get_plugin_manager
)

# Experiment Tracking
EXPERIMENT_TRACKING_AVAILABLE = False
try:
    from .core.experiments.experiment_tracker import (
        ExperimentTracker,
        create_tracker
    )
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError:
    pass

# Data Processing
from .core.data.datasets.recovery_dataset import (
    RecoveryDataset,
    SequenceDataset,
    TextDataset,
    create_recovery_dataset,
    create_sequence_dataset,
    create_text_dataset
)

# Evaluation
from .core.evaluation.metrics import (
    MetricsCalculator,
    ModelEvaluator,
    create_evaluator
)

# Checkpointing
from .core.checkpointing.checkpoint_manager import (
    CheckpointManager,
    create_checkpoint_manager
)

# Ultra-Fast Optimization
from .core.optimization.ultra_fast_inference import (
    UltraFastInference,
    AsyncInferenceEngine,
    EmbeddingCache,
    BatchOptimizer,
    create_ultra_fast_inference,
    create_async_engine,
    create_embedding_cache
)

# Memory Optimization
from .core.optimization.memory_optimizer import (
    MemoryOptimizer,
    GradientCheckpointing,
    optimize_model_memory,
    clear_memory_cache,
    get_memory_stats
)

# Pipeline Optimization
from .core.optimization.pipeline_optimizer import (
    InferencePipeline,
    StreamingInference,
    create_inference_pipeline,
    create_streaming_inference
)

__all__ = [
    "InputValidator",
    "ModelValidator",
    "validate_input",
    "validate_features",
    "validate_text",
    "ModelTester",
    "create_model_tester",
    "SystemHealthMonitor",
    "ModelHealthMonitor",
    "create_system_monitor",
    "create_model_monitor",
    "RecoveryAIError",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "ModelTrainingError",
    "DataError",
    "DataValidationError",
    "DataProcessingError",
    "ConfigurationError",
    "InferenceError",
    "CUDAOutOfMemoryError",
    "ValidationError",
    "ErrorHandler",
    "handle_errors",
    "safe_inference",
    "IntegratedPipeline",
    "create_integrated_pipeline",
    "StructuredFormatter",
    "ModelLogger",
    "create_model_logger",
    "ModelBenchmark",
    "create_benchmark",
    "PerformanceOptimizer",
    "enable_optimizations",
    "profile_model",
    "init_weights_xavier",
    "init_weights_kaiming",
    "init_weights_orthogonal",
    "init_weights_normal",
    "init_weights_zero",
    "initialize_model",
    "TrainingVisualizer",
    "ModelVisualizer",
    "create_training_visualizer",
    "create_model_visualizer",
    "ModelSecurity",
    "compute_model_hash",
    "verify_model_integrity",
    "sanitize_input",
    "SmartCache",
    "create_smart_cache",
    "AdvancedMetrics",
    "calculate_regression_metrics",
    "calculate_classification_metrics",
    "calculate_correlation",
    "ModelSerializer",
    "save_model",
    "load_model",
    "ModelExporter",
    "export_to_onnx",
    "export_to_torchscript",
    "get_device",
    "count_parameters",
    "freeze_layers",
    "unfreeze_layers",
    "set_learning_rate",
    "get_learning_rate",
    "format_number",
    "DeploymentConfig",
    "HealthCheck",
    "get_deployment_config",
    "check_system_health",
    "check_model_health",
    "ModelFactory",
    "ModelBuilder",
    "TrainerFactory",
    "ConfigLoader",
    "get_config",
    "DataLoaderFactory",
    "Plugin",
    "PluginManager",
    "get_plugin_manager",
    "EXPERIMENT_TRACKING_AVAILABLE",
    "RecoveryDataset",
    "SequenceDataset",
    "TextDataset",
    "create_recovery_dataset",
    "create_sequence_dataset",
    "create_text_dataset",
    "MetricsCalculator",
    "ModelEvaluator",
    "create_evaluator",
    "CheckpointManager",
    "create_checkpoint_manager",
    "UltraFastInference",
    "AsyncInferenceEngine",
    "EmbeddingCache",
    "BatchOptimizer",
    "create_ultra_fast_inference",
    "create_async_engine",
    "create_embedding_cache",
    "MemoryOptimizer",
    "GradientCheckpointing",
    "optimize_model_memory",
    "clear_memory_cache",
    "get_memory_stats",
    "InferencePipeline",
    "StreamingInference",
    "create_inference_pipeline",
    "create_streaming_inference",
]

# Conditionally add experiment tracking
if EXPERIMENT_TRACKING_AVAILABLE:
    __all__.extend([
        "ExperimentTracker",
        "create_tracker",
    ])


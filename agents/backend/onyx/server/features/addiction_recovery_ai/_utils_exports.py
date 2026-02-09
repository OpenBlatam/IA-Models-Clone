"""
Utility exports for Addiction Recovery AI
"""

# Debugging Tools
from .utils.debugging_tools import (
    ModelDebugger,
    PerformanceProfiler,
    TrainingMonitor,
    create_model_debugger,
    create_profiler,
    create_training_monitor
)

# Advanced Gradio
ADVANCED_GRADIO_AVAILABLE = False
try:
    from .utils.advanced_gradio import (
        AdvancedRecoveryGradio,
        create_advanced_gradio_app
    )
    ADVANCED_GRADIO_AVAILABLE = True
except ImportError:
    pass

# Utilities
from .utils.model_utils import (
    count_parameters,
    get_model_size,
    freeze_model,
    unfreeze_model,
    freeze_layers,
    get_layer_output_shape,
    compare_models,
    export_model,
    load_model_from_checkpoint,
    transfer_weights,
    get_layer_info,
    save_model_checkpoint,
    load_model_checkpoint
)

from .utils.data_utils import (
    normalize_features,
    split_data,
    create_sequences,
    augment_data,
    balance_dataset,
    create_cross_validation_splits
)

# Gradio Interface
from .utils.gradio_recovery import RecoveryGradioInterface, create_recovery_gradio_app

# Advanced features
from .core.models.onnx_optimizer import (
    export_to_onnx, optimize_onnx, ONNXRuntimeInference
)
from .core.models.model_optimization import (
    ModelPruner, KnowledgeDistillation, create_lightweight_student
)
from .utils.enhanced_gradio import EnhancedRecoveryGradio, create_enhanced_gradio_app
from .utils.error_handler import (
    ErrorHandler, RecoveryAIError, ModelLoadError, InferenceError, CUDAOutOfMemoryError
)
from .utils.precomputation import (
    EmbeddingCache, FeaturePreprocessor, BatchPreprocessor, precompute_embeddings
)
from .utils.async_inference import (
    AsyncInferenceEngine, InferenceQueue, ParallelInference
)
from .utils.profiler import (
    PerformanceProfiler, ModelProfiler, SystemMonitor
)
from .utils.data_augmentation import (
    FeatureAugmentation, SequenceAugmentation, AugmentationPipeline,
    create_feature_augmentation_pipeline, create_sequence_augmentation_pipeline
)
from .utils.advanced_logging import (
    StructuredLogger, MetricsTracker, TrainingLogger, setup_logging,
    JSONFormatter, LogAggregator
)
from .core.models.model_ensembling import (
    ModelEnsemble, StackingEnsemble, create_ensemble, create_stacking_ensemble
)
from .utils.hyperparameter_optimization import (
    HyperparameterOptimizer, GridSearchOptimizer
)
from .utils.model_versioning import (
    ModelVersion, ModelRegistry, ModelComparator
)
from .utils.ab_testing import (
    ABTest, MultiVariateTest
)
from .utils.model_interpretability import (
    ModelInterpreter, AttentionVisualizer
)
from .utils.continuous_learning import (
    OnlineLearner, IncrementalLearner, AdaptiveLearningRate
)
from .utils.automl import (
    AutoML, NeuralArchitectureSearch
)
from .utils.model_serving import (
    ModelServer, ModelCache, RateLimiter, retry_on_failure, HealthMonitor
)
from .utils.advanced_validation import (
    DataValidator, AnomalyDetector, DataQualityChecker
)
from .utils.advanced_compression import (
    AdvancedQuantization, ModelCompressor, TensorRTOptimizer
)
from .utils.security import (
    APIKeyManager, InputSanitizer, RateLimiter as SecurityRateLimiter, SecureHash
)
from .config.config_manager import ConfigManager, load_config
from .utils.metrics_dashboard import MetricsDashboard, PerformanceTracker
from .utils.experiment_tracking import (
    ExperimentTracker, TrainingLogger as ExperimentTrainingLogger
)
from .utils.visualization import (
    ModelVisualizer, ProgressVisualizer
)
from .utils.data_pipeline import (
    RecoveryDataset, SequenceDataset, DataPipeline, DataAugmentationPipeline
)
from .utils.streaming import (
    StreamProcessor, RealTimePredictor
)
from .utils.distributed_inference import (
    DistributedInference
)
from .utils.advanced_caching import (
    LRUCache, PersistentCache, CacheDecorator
)
from .utils.multi_tenant import (
    TenantManager, TenantIsolation
)
from .utils.backup_recovery import (
    ModelBackup, DataBackup
)
from .utils.monitoring_dashboard import (
    SystemMonitor, PerformanceMonitor
)
from .utils.error_recovery import (
    CircuitBreaker, RetryHandler, GracefulDegradation
)
from .utils.analytics import (
    RecoveryAnalytics, CohortAnalysis
)
from .utils.notifications import (
    NotificationManager, NotificationLevel, AlertSystem
)
from .utils.integration import (
    ExternalAPIClient, WebhookHandler, DataExporter
)
from .utils.load_balancer import (
    LoadBalancer, ModelPool
)
from .utils.feature_store import (
    FeatureStore, EmbeddingStore
)
from .utils.scheduler import (
    TaskScheduler, ModelUpdateScheduler
)
from .api.graphql_api import GraphQLAPI
from .utils.message_queue import (
    MessageQueue, EventStream
)
from .utils.api_versioning import (
    APIVersion, VersionRouter, versioned
)
from .utils.service_discovery import (
    ServiceRegistry, ServiceDiscovery
)
from .utils.benchmarking import (
    Benchmark, ModelBenchmark
)
from .utils.advanced_testing import (
    ModelTester, DataTester, MockFactory
)
from .utils.documentation_generator import (
    DocumentationGenerator
)
from .utils.resource_manager import (
    ResourceMonitor, MemoryManager, ResourceLimiter
)
from .utils.advanced_config import (
    ConfigManager, ModelConfig, TrainingConfig, InferenceConfig
)
from .utils.advanced_pipeline import (
    ProcessingPipeline, PipelineStage, BatchProcessor
)
from .utils.session_manager import (
    SessionManager, Session
)
from .utils.rate_limiter_advanced import (
    AdvancedRateLimiter, TokenBucket, SlidingWindowLimiter
)

__all__ = [
    "ModelDebugger",
    "PerformanceProfiler",
    "TrainingMonitor",
    "create_model_debugger",
    "create_profiler",
    "create_training_monitor",
    "ADVANCED_GRADIO_AVAILABLE",
    "count_parameters",
    "get_model_size",
    "freeze_model",
    "unfreeze_model",
    "freeze_layers",
    "get_layer_output_shape",
    "compare_models",
    "export_model",
    "load_model_from_checkpoint",
    "transfer_weights",
    "get_layer_info",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "normalize_features",
    "split_data",
    "create_sequences",
    "augment_data",
    "balance_dataset",
    "create_cross_validation_splits",
    "RecoveryGradioInterface",
    "create_recovery_gradio_app",
    "export_to_onnx",
    "optimize_onnx",
    "ONNXRuntimeInference",
    "ModelPruner",
    "KnowledgeDistillation",
    "create_lightweight_student",
    "EnhancedRecoveryGradio",
    "create_enhanced_gradio_app",
    "ErrorHandler",
    "RecoveryAIError",
    "ModelLoadError",
    "InferenceError",
    "CUDAOutOfMemoryError",
    "EmbeddingCache",
    "FeaturePreprocessor",
    "BatchPreprocessor",
    "precompute_embeddings",
    "AsyncInferenceEngine",
    "InferenceQueue",
    "ParallelInference",
    "ModelProfiler",
    "SystemMonitor",
    "FeatureAugmentation",
    "SequenceAugmentation",
    "AugmentationPipeline",
    "create_feature_augmentation_pipeline",
    "create_sequence_augmentation_pipeline",
    "StructuredLogger",
    "MetricsTracker",
    "TrainingLogger",
    "setup_logging",
    "JSONFormatter",
    "LogAggregator",
    "ModelEnsemble",
    "StackingEnsemble",
    "create_ensemble",
    "create_stacking_ensemble",
    "HyperparameterOptimizer",
    "GridSearchOptimizer",
    "ModelVersion",
    "ModelRegistry",
    "ModelComparator",
    "ABTest",
    "MultiVariateTest",
    "ModelInterpreter",
    "AttentionVisualizer",
    "OnlineLearner",
    "IncrementalLearner",
    "AdaptiveLearningRate",
    "AutoML",
    "NeuralArchitectureSearch",
    "ModelServer",
    "ModelCache",
    "RateLimiter",
    "retry_on_failure",
    "HealthMonitor",
    "DataValidator",
    "AnomalyDetector",
    "DataQualityChecker",
    "AdvancedQuantization",
    "ModelCompressor",
    "TensorRTOptimizer",
    "APIKeyManager",
    "InputSanitizer",
    "SecurityRateLimiter",
    "SecureHash",
    "ConfigManager",
    "load_config",
    "MetricsDashboard",
    "PerformanceTracker",
    "ExperimentTracker",
    "ExperimentTrainingLogger",
    "ModelVisualizer",
    "ProgressVisualizer",
    "RecoveryDataset",
    "SequenceDataset",
    "DataPipeline",
    "DataAugmentationPipeline",
    "StreamProcessor",
    "RealTimePredictor",
    "DistributedInference",
    "LRUCache",
    "PersistentCache",
    "CacheDecorator",
    "TenantManager",
    "TenantIsolation",
    "ModelBackup",
    "DataBackup",
    "PerformanceMonitor",
    "CircuitBreaker",
    "RetryHandler",
    "GracefulDegradation",
    "RecoveryAnalytics",
    "CohortAnalysis",
    "NotificationManager",
    "NotificationLevel",
    "AlertSystem",
    "ExternalAPIClient",
    "WebhookHandler",
    "DataExporter",
    "LoadBalancer",
    "ModelPool",
    "FeatureStore",
    "EmbeddingStore",
    "TaskScheduler",
    "ModelUpdateScheduler",
    "GraphQLAPI",
    "MessageQueue",
    "EventStream",
    "APIVersion",
    "VersionRouter",
    "versioned",
    "ServiceRegistry",
    "ServiceDiscovery",
    "Benchmark",
    "ModelBenchmark",
    "ModelTester",
    "DataTester",
    "MockFactory",
    "DocumentationGenerator",
    "ResourceMonitor",
    "MemoryManager",
    "ResourceLimiter",
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "ProcessingPipeline",
    "PipelineStage",
    "BatchProcessor",
    "SessionManager",
    "Session",
    "AdvancedRateLimiter",
    "TokenBucket",
    "SlidingWindowLimiter",
]

# Conditionally add advanced gradio
if ADVANCED_GRADIO_AVAILABLE:
    __all__.extend([
        "AdvancedRecoveryGradio",
        "create_advanced_gradio_app",
    ])


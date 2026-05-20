"""
Shared ML Module
Comprehensive utilities for machine learning operations.
Ultra-modular architecture with specialized components.
"""

# Core interfaces and factories
from .core import (
    IModelLoader,
    IInferenceEngine,
    ITrainer,
    IEvaluator,
    ModelLoaderFactory,
    OptimizerFactory,
    LossFunctionFactory,
    DeviceFactory,
    ComponentFactory,
    TrainingPipelineBuilder,
    InferencePipelineBuilder,
    ModelOptimizationBuilder,
    FocalLoss,
    LabelSmoothingLoss,
    DiceLoss,
    ContrastiveLoss,
)

# Configuration
from .config import (
    ModelConfig,
    TrainingConfig,
    LoRAConfig,
    GenerationConfig,
    DiffusionConfig,
    MLServiceSettings,
    load_config,
    save_config,
    DEFAULT_CONFIG,
)

# Model utilities
from .model_utils import (
    get_device,
    get_dtype,
    count_parameters,
    estimate_model_size_mb,
    load_model_checkpoint,
    save_model_checkpoint,
    enable_mixed_precision,
    freeze_model_parameters,
    get_model_summary,
    clip_gradients,
    check_for_nan_inf,
)

# Data utilities
from .data_utils import (
    TextDataset,
    create_dataloader,
    split_dataset,
    collate_fn_padding,
    normalize_text,
    create_vocab_from_texts,
)

# Base models
from .models.base_model import BaseLLMModel, ModelManager

# Data loading
from .data.data_loader import (
    TextDataset as FunctionalTextDataset,
    create_dataloader as create_optimized_dataloader,
    split_dataset as split_dataset_functional,
    collate_fn_padding as collate_with_padding,
    create_data_pipeline,
)

# Training
from .training.trainer import Trainer

# Evaluation
from .evaluation.evaluator import Evaluator

# Experiment tracking
from .tracking.experiment_tracker import ExperimentTracker

# Inference
from .inference.inference_engine import InferenceEngine
from .inference.batch_processor import BatchProcessor, DynamicBatchProcessor

# Optimization
from .optimization.lora_manager import LoRAManager
from .optimization.optimizers import (
    OptimizerWithWarmup,
    LookaheadOptimizer,
    create_optimizer_with_schedule,
)

# Schedulers
from .schedulers.learning_rate_scheduler import (
    LearningRateSchedulerFactory,
    EarlyStopping,
)

# Distributed training
from .distributed.distributed_trainer import DistributedTrainer

# Quantization
from .quantization.quantization_manager import QuantizationManager

# Monitoring and profiling
from .monitoring.profiler import ModelProfiler, BottleneckAnalyzer

# Model registry
from .registry.model_registry import ModelRegistry, ModelVersion

# Gradio utilities
from .gradio.gradio_utils import (
    gradio_error_handler,
    create_text_generation_interface,
    create_image_generation_interface,
    create_chat_interface,
    create_batch_interface,
)

# Utilities
from .utils import (
    timing_decorator,
    gpu_memory_tracker,
    error_handler,
    validate_inputs,
    cache_result,
    retry,
    torch_no_grad,
    torch_eval_mode,
    validate_model_input,
    validate_generation_params,
    DataTransformer,
    ComposeTransformer,
    create_text_transformer_pipeline,
    create_image_transformer_pipeline,
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LoggingCallback,
    CallbackManager,
)

# Adapters
from .adapters import (
    ModelAdapter,
    HuggingFaceAdapter,
    ONNXAdapter,
    TensorFlowAdapter,
    AdapterRegistry,
)

# Plugins
from .plugins import (
    Plugin,
    ModelPlugin,
    DataPlugin,
    TrainingPlugin,
    PluginManager,
)

# Composition
from .composition import (
    PipelineStage,
    PipelineComposer,
    TrainingPipelineComposer,
    InferencePipelineComposer,
)

# Strategies
from .strategies import (
    OptimizationStrategy,
    LoRAStrategy,
    QuantizationStrategy,
    PruningStrategy,
    OptimizationContext,
    TrainingStrategy,
    StandardTrainingStrategy,
    DistributedTrainingStrategy,
    TrainingContext,
)

# Events
from .events import (
    EventType,
    Event,
    EventListener,
    EventEmitter,
    LoggingEventListener,
    MetricsEventListener,
    EventBus,
)

# Middleware
from .middleware import (
    Middleware,
    LoggingMiddleware,
    ValidationMiddleware,
    TimingMiddleware,
    CachingMiddleware,
    MiddlewarePipeline,
    MiddlewareManager,
)

# Services
from .services import (
    BaseService,
    ModelService,
    InferenceService,
    TrainingService,
    ServiceRegistry,
)

# Repositories
from .repositories import (
    Repository,
    ModelRepository,
    CheckpointRepository,
    ConfigRepository,
    RepositoryManager,
)

# Specialized modules
from .models.transformer import (
    TransformerBlock,
    CausalTransformerModel,
)

from .data.preprocessing import (
    TextPreprocessor,
    ImagePreprocessor,
    create_text_preprocessor,
    create_image_preprocessor,
)

from .training.callbacks import (
    GradientMonitorCallback,
    LearningRateMonitorCallback,
    TrainingModelCheckpointCallback,
)

from .evaluation.metrics import (
    MetricCalculator,
    MetricsAggregator,
)

# Cache
from .cache import (
    CacheEntry,
    LRUCache,
    CacheManager,
)

# Security
from .security import (
    InputSanitizer,
    RateLimiter,
    ResourceLimiter,
)

# Performance
from .performance import (
    ModelOptimizer,
    MemoryOptimizer,
)

# Async Operations
from .async_ops import (
    AsyncExecutor,
    AsyncModelInference,
    asyncify,
)

# Integration
from .integration import (
    ServiceOrchestrator,
    PipelineBuilder,
)

# Streaming
from .streaming import (
    StreamProcessor,
    TokenStreamer,
    ImageStreamer,
)

# Validation
from .validation import (
    ModelInputValidator,
    ImageValidator,
    ConfigValidator,
    ValidatorChain,
)

# Health
from .health import (
    HealthStatus,
    ComponentHealth,
    HealthChecker,
)

# Metrics/Telemetry
from .metrics import (
    Metric,
    Counter,
    Gauge,
    Histogram,
    TelemetryCollector,
    get_collector,
)

# Retry
from .retry import (
    RetryConfig,
    RetryHandler,
    CircuitBreaker,
    retry,
)

# Batch
from .batch import (
    Priority,
    BatchItem,
    BatchManager,
    DynamicBatchProcessor,
)

# Compression
from .compression import (
    Compressor,
    ModelCompressor,
)

# Backup
from .backup import (
    BackupManager,
)

# Error handling
from .errors import (
    MLServiceError,
    ModelLoadError,
    ModelNotFoundError,
    InferenceError,
    TrainingError,
    ConfigurationError,
    DataError,
    GPUError,
)

__all__ = [
    # Core
    "IModelLoader",
    "IInferenceEngine",
    "ITrainer",
    "IEvaluator",
    "ModelLoaderFactory",
    "OptimizerFactory",
    "LossFunctionFactory",
    "DeviceFactory",
    "ComponentFactory",
    "TrainingPipelineBuilder",
    "InferencePipelineBuilder",
    "ModelOptimizationBuilder",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "ContrastiveLoss",
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "LoRAConfig",
    "GenerationConfig",
    "DiffusionConfig",
    "MLServiceSettings",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG",
    # Model utilities
    "get_device",
    "get_dtype",
    "count_parameters",
    "estimate_model_size_mb",
    "load_model_checkpoint",
    "save_model_checkpoint",
    "enable_mixed_precision",
    "freeze_model_parameters",
    "get_model_summary",
    "clip_gradients",
    "check_for_nan_inf",
    # Models
    "BaseLLMModel",
    "ModelManager",
    "TransformerBlock",
    "CausalTransformerModel",
    # Data
    "TextDataset",
    "FunctionalTextDataset",
    "create_dataloader",
    "create_optimized_dataloader",
    "split_dataset",
    "split_dataset_functional",
    "collate_fn_padding",
    "collate_with_padding",
    "create_data_pipeline",
    "normalize_text",
    "create_vocab_from_texts",
    "TextPreprocessor",
    "ImagePreprocessor",
    "create_text_preprocessor",
    "create_image_preprocessor",
    # Training
    "Trainer",
    "GradientMonitorCallback",
    "LearningRateMonitorCallback",
    "TrainingModelCheckpointCallback",
    # Evaluation
    "Evaluator",
    "MetricCalculator",
    "MetricsAggregator",
    # Tracking
    "ExperimentTracker",
    # Inference
    "InferenceEngine",
    "BatchProcessor",
    "DynamicBatchProcessor",
    # Optimization
    "LoRAManager",
    "OptimizerWithWarmup",
    "LookaheadOptimizer",
    "create_optimizer_with_schedule",
    # Schedulers
    "LearningRateSchedulerFactory",
    "EarlyStopping",
    # Distributed
    "DistributedTrainer",
    # Quantization
    "QuantizationManager",
    # Monitoring
    "ModelProfiler",
    "BottleneckAnalyzer",
    # Registry
    "ModelRegistry",
    "ModelVersion",
    # Gradio
    "gradio_error_handler",
    "create_text_generation_interface",
    "create_image_generation_interface",
    "create_chat_interface",
    "create_batch_interface",
    # Utilities
    "timing_decorator",
    "gpu_memory_tracker",
    "error_handler",
    "validate_inputs",
    "cache_result",
    "retry",
    "torch_no_grad",
    "torch_eval_mode",
    "validate_model_input",
    "validate_generation_params",
    "DataTransformer",
    "ComposeTransformer",
    "create_text_transformer_pipeline",
    "create_image_transformer_pipeline",
    "Callback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LoggingCallback",
    "CallbackManager",
    # Adapters
    "ModelAdapter",
    "HuggingFaceAdapter",
    "ONNXAdapter",
    "TensorFlowAdapter",
    "AdapterRegistry",
    # Plugins
    "Plugin",
    "ModelPlugin",
    "DataPlugin",
    "TrainingPlugin",
    "PluginManager",
    # Composition
    "PipelineStage",
    "PipelineComposer",
    "TrainingPipelineComposer",
    "InferencePipelineComposer",
    # Strategies
    "OptimizationStrategy",
    "LoRAStrategy",
    "QuantizationStrategy",
    "PruningStrategy",
    "OptimizationContext",
    "TrainingStrategy",
    "StandardTrainingStrategy",
    "DistributedTrainingStrategy",
    "TrainingContext",
    # Events
    "EventType",
    "Event",
    "EventListener",
    "EventEmitter",
    "LoggingEventListener",
    "MetricsEventListener",
    "EventBus",
    # Middleware
    "Middleware",
    "LoggingMiddleware",
    "ValidationMiddleware",
    "TimingMiddleware",
    "CachingMiddleware",
    "MiddlewarePipeline",
    "MiddlewareManager",
    # Services
    "BaseService",
    "ModelService",
    "InferenceService",
    "TrainingService",
    "ServiceRegistry",
    # Repositories
    "Repository",
    "ModelRepository",
    "CheckpointRepository",
    "ConfigRepository",
    "RepositoryManager",
    # Cache
    "CacheEntry",
    "LRUCache",
    "CacheManager",
    # Security
    "InputSanitizer",
    "RateLimiter",
    "ResourceLimiter",
    # Performance
    "ModelOptimizer",
    "MemoryOptimizer",
    # Async Operations
    "AsyncExecutor",
    "AsyncModelInference",
    "asyncify",
    # Integration
    "ServiceOrchestrator",
    "PipelineBuilder",
    # Streaming
    "StreamProcessor",
    "TokenStreamer",
    "ImageStreamer",
    # Validation
    "ModelInputValidator",
    "ImageValidator",
    "ConfigValidator",
    "ValidatorChain",
    # Health
    "HealthStatus",
    "ComponentHealth",
    "HealthChecker",
    # Metrics/Telemetry
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "TelemetryCollector",
    "get_collector",
    # Retry
    "RetryConfig",
    "RetryHandler",
    "CircuitBreaker",
    "retry",
    # Batch
    "Priority",
    "BatchItem",
    "BatchManager",
    "DynamicBatchProcessor",
    # Compression
    "Compressor",
    "ModelCompressor",
    # Backup
    "BackupManager",
    # Errors
    "MLServiceError",
    "ModelLoadError",
    "ModelNotFoundError",
    "InferenceError",
    "TrainingError",
    "ConfigurationError",
    "DataError",
    "GPUError",
]

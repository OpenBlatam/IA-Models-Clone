"""
Music Analyzer AI - Ultra-Modular Deep Learning System
=======================================================

Enhanced with modular architecture, registry, and composition systems.

Features:
- Advanced transformer models for music analysis
- Multi-task learning capabilities
- Modular architecture with composition patterns
- Comprehensive training and inference pipelines
- Experiment tracking and monitoring
- Model optimization and export utilities
"""

__version__ = "4.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Ultra-modular deep learning system for music analysis"

# Core systems
from .core.registry import (
    get_registry,
    register_model,
    register_loss,
    register_optimizer,
    register_scheduler
)
from .core.composition import (
    ModelComposer,
    SequentialComposer,
    ParallelComposer
)
from .core.model_manager import ModelManager

# Factories
from .factories.unified_factory import get_factory, UnifiedFactory

# Configuration
from .config.model_config import (
    ModelConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    ConfigManager
)

# Model architectures (modular components)
from .models.architectures import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    AttentionLayer,
    LayerNorm,
    BatchNorm1d,
    AdaptiveNormalization,
    FeedForward,
    GatedFeedForward,
    ResidualFeedForward,
    PositionalEncoding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    FeatureEmbedding,
    AudioFeatureEmbedding,
    MusicFeatureEmbedding,
    GELU,
    Swish,
    Mish,
    GLU,
    ActivationFactory,
    create_activation,
    MeanPooling,
    MaxPooling,
    AttentionPooling,
    AdaptivePooling,
    PoolingFactory,
    create_pooling
)

# Attention submodules
from .models.architectures.attention import (
    ScaledDotProductAttention as ScaledDotProductAttentionV2,
    MultiHeadAttention as MultiHeadAttentionV2
)

# Loss submodules
from .training.components.losses import (
    ClassificationLoss as ClassificationLossV2,
    FocalLoss as FocalLossV2,
    LabelSmoothingLoss as LabelSmoothingLossV2,
    RegressionLoss as RegressionLossV2
)

# Evaluation metrics submodules
from .evaluation.metrics import (
    ClassificationMetrics as ClassificationMetricsV2,
    RegressionMetrics as RegressionMetricsV2
)

# Inference pipelines
from .inference.pipelines.batch_pipeline import BatchInferencePipeline

# Builders
from .builders.model_builder import ModelBuilder

# Modular models
from .models.modular_transformer import (
    ModularTransformerEncoder,
    ModularMusicClassifier,
    TransformerEncoderLayer
)

# Training components
from .training.components import (
    ClassificationLoss,
    RegressionLoss,
    MultiTaskLoss,
    FocalLoss,
    LabelSmoothingLoss,
    OptimizerFactory,
    create_optimizer,
    SchedulerFactory,
    create_scheduler,
    WarmupScheduler,
    TrainingCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LearningRateCallback,
    MetricsCallback
)

# Training loops
from .training.loops import (
    BaseTrainingLoop,
    StandardTrainingLoop
)

# Training strategies
from .training.strategies import (
    BaseTrainingStrategy,
    StandardTrainingStrategy,
    MixedPrecisionStrategy,
    DistributedTrainingStrategy
)

# Inference pipelines
from .inference.pipelines import (
    BaseInferencePipeline,
    StandardInferencePipeline
)

# Data transformations
from .data.transforms import (
    AudioNormalizer,
    AudioResampler,
    AudioTrimmer,
    AudioPadder,
    AudioAugmenter,
    FeatureNormalizer,
    FeatureScaler,
    FeatureSelector,
    FeatureCombiner,
    Compose,
    ComposeTransforms
)

# Data augmentations
from .data.augmentations import (
    TimeStretchAugmentation,
    PitchShiftAugmentation,
    NoiseAugmentation,
    VolumeAugmentation,
    TimeMaskAugmentation,
    FrequencyMaskAugmentation,
    FeatureNoiseAugmentation,
    FeatureScaleAugmentation,
    FeatureShiftAugmentation,
    AugmentationFactory,
    create_augmentation
)

# Data pipelines
from .data.pipelines import (
    FeatureExtractionPipeline,
    create_standard_feature_pipeline
)

# Integrations
# Integrations - try new submodules first, fallback to old
try:
    from .integrations.transformers import EnhancedTransformerWrapper
    try:
        from .integrations.transformers import LoRATransformerWrapper
    except ImportError:
        LoRATransformerWrapper = None
except ImportError:
    EnhancedTransformerWrapper = None
    LoRATransformerWrapper = None

try:
    from .integrations.diffusion import DiffusionSchedulerFactory, DiffusionPipelineWrapper
except ImportError:
    DiffusionSchedulerFactory = None
    DiffusionPipelineWrapper = None

# Backward compatibility
try:
    from .integrations.transformers_integration import (
        HuggingFaceModelWrapper,
        TransformerMusicEncoder
    )
except ImportError:
    HuggingFaceModelWrapper = None
    TransformerMusicEncoder = None

# Evaluation
from .evaluation.modular_metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    MultiTaskMetrics
)

# Gradio components
from .gradio.components import (
    ModelInferenceComponent,
    VisualizationComponent
)

# Checkpoint management
from .checkpoints import (
    CheckpointManager,
    CheckpointLoader,
    CheckpointSaver,
    CheckpointValidator
)

# Experiment tracking
from .experiments.trackers import (
    BaseExperimentTracker,
    WandBTracker,
    TensorBoardTracker,
    MLflowTracker,
    TrackerFactory,
    create_tracker
)

# Optimization utilities
from .optimization import (
    GradientClipper,
    GradientAccumulator,
    GradientMonitor,
    LearningRateFinder,
    LearningRateScheduler,
    WarmupScheduler,
    ModelQuantizer,
    ModelPruner,
    ModelCompressor
)

# Debugging utilities
from .debugging import (
    TrainingDebugger,
    InferenceDebugger,
    GradientDebugger,
    NaNDetector
)

# Monitoring utilities
from .monitoring import (
    PerformanceMonitor,
    MemoryMonitor,
    TrainingMonitor,
    ModelMonitor
)

# Profiling utilities
from .profiling import (
    ModelProfiler,
    TrainingProfiler,
    InferenceProfiler
)

# Serialization utilities
from .serialization import (
    ModelSerializer,
    ConfigSerializer,
    DataSerializer
)

# Logging utilities
from .logging import (
    LoggerFactory,
    create_logger,
    TrainingLogger,
    InferenceLogger,
    MetricsLogger
)

# Visualization utilities
from .visualization import (
    MetricsPlotter,
    TrainingPlotter,
    ModelVisualizer,
    ConfusionMatrixPlotter
)

# Export utilities
from .export import (
    ModelExporter,
    ONNXExporter,
    TorchScriptExporter
)

# Enhanced core components
from .core.device_context import DeviceContext, TrainingContext

# Enhanced training components
from .training.strategies.enhanced_mixed_precision import EnhancedMixedPrecisionStrategy
from .training.data_loader_enhanced import EnhancedDataLoader, SmartBatchSampler

# Enhanced debugging
from .debugging.gradient_analyzer import GradientAnalyzer, GradientMonitor

# Enhanced integrations
# EnhancedTransformerWrapper already imported above from .integrations.transformers

# Utilities
from .utils.device_manager import get_device_manager, DeviceManager
from .utils.initialization import initialize_weights, WeightInitializer
from .utils.validation import (
    TensorValidator,
    ArrayValidator,
    InputValidator
)

# Interfaces
from .interfaces.base import (
    IModel,
    IEmbeddingModel,
    IClassifier,
    ITrainingStrategy,
    ILossFunction,
    IOptimizer,
    IScheduler,
    IDataTransform,
    IDataAugmentation,
    ITrainingCallback,
    IInferencePipeline,
    IMonitor,
    IProfiler,
    ICheckpointManager,
    IExperimentTracker,
    IFactory,
    IDeviceManager,
)

# Specialized factories
from .factories.model_factory import ModelFactory
from .factories.training_factory import TrainingFactory

# Training executors
from .training.executors.base_executor import (
    BaseTrainingExecutor,
    StandardTrainingExecutor
)

# Data loaders
from .data.loaders.base_loader import (
    BaseDataset,
    BaseDataLoaderFactory,
    StandardDataLoaderFactory
)

# Original exports (for backward compatibility)
# Import from core.models submodule (backward compatibility)
try:
    from .core.models import (
        DeepGenreClassifier,
        DeepMoodDetector,
        MultiTaskMusicModel,
        TransformerMusicEncoder as OriginalTransformerEncoder,
        DeepMusicAnalyzer,
        get_deep_analyzer
    )
except ImportError:
    # Fallback to old location
    from .core.deep_models import (
        DeepGenreClassifier,
        DeepMoodDetector,
        MultiTaskMusicModel,
        TransformerMusicEncoder as OriginalTransformerEncoder,
        DeepMusicAnalyzer,
        get_deep_analyzer
    )

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core systems
    "get_registry",
    "register_model",
    "register_loss",
    "register_optimizer",
    "register_scheduler",
    "ModelComposer",
    "SequentialComposer",
    "ParallelComposer",
    "ModelManager",
    # Factories
    "get_factory",
    "UnifiedFactory",
    # Configuration
    "ModelConfig",
    "ModelArchitectureConfig",
    "TrainingConfig",
    "DataConfig",
    "ExperimentConfig",
    "ConfigManager",
    # Model architectures
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "AttentionLayer",
    "LayerNorm",
    "BatchNorm1d",
    "AdaptiveNormalization",
    "FeedForward",
    "GatedFeedForward",
    "ResidualFeedForward",
    "PositionalEncoding",
    "LearnedPositionalEncoding",
    "SinusoidalPositionalEncoding",
    "FeatureEmbedding",
    "AudioFeatureEmbedding",
    "MusicFeatureEmbedding",
    "GELU",
    "Swish",
    "Mish",
    "GLU",
    "ActivationFactory",
    "create_activation",
    "MeanPooling",
    "MaxPooling",
    "AttentionPooling",
    "AdaptivePooling",
    "PoolingFactory",
    "create_pooling",
    # Dropout
    "StandardDropout",
    "SpatialDropout",
    "AlphaDropout",
    "DropoutFactory",
    "create_dropout",
    # Residual
    "ResidualConnection",
    "PreNormResidual",
    "PostNormResidual",
    "GatedResidual",
    # Modular models
    "ModularTransformerEncoder",
    "ModularMusicClassifier",
    "TransformerEncoderLayer",
    # Training components
    "ClassificationLoss",
    "RegressionLoss",
    "MultiTaskLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "OptimizerFactory",
    "create_optimizer",
    "SchedulerFactory",
    "create_scheduler",
    "WarmupScheduler",
    "TrainingCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LearningRateCallback",
    "MetricsCallback",
    # Training loops
    "BaseTrainingLoop",
    "StandardTrainingLoop",
    # Training strategies
    "BaseTrainingStrategy",
    "StandardTrainingStrategy",
    "MixedPrecisionStrategy",
    "DistributedTrainingStrategy",
    # Inference pipelines
    "BaseInferencePipeline",
    "StandardInferencePipeline",
    # Data transformations
    "AudioNormalizer",
    "AudioResampler",
    "AudioTrimmer",
    "AudioPadder",
    "AudioAugmenter",
    "FeatureNormalizer",
    "FeatureScaler",
    "FeatureSelector",
    "FeatureCombiner",
    "Compose",
    "ComposeTransforms",
    # Data augmentations
    "TimeStretchAugmentation",
    "PitchShiftAugmentation",
    "NoiseAugmentation",
    "VolumeAugmentation",
    "TimeMaskAugmentation",
    "FrequencyMaskAugmentation",
    "FeatureNoiseAugmentation",
    "FeatureScaleAugmentation",
    "FeatureShiftAugmentation",
    "AugmentationFactory",
    "create_augmentation",
    # Data pipelines
    "FeatureExtractionPipeline",
    "create_standard_feature_pipeline",
    # Integrations
    "HuggingFaceModelWrapper",
    "TransformerMusicEncoder",
    "LoRATransformerWrapper",
    "DiffusionSchedulerFactory",
    "DiffusionPipelineWrapper",
    # Evaluation
    "ClassificationMetrics",
    "RegressionMetrics",
    "MultiTaskMetrics",
    # Gradio components
    "ModelInferenceComponent",
    "VisualizationComponent",
    # Checkpoint management
    "CheckpointManager",
    "CheckpointLoader",
    "CheckpointSaver",
    "CheckpointValidator",
    # Experiment tracking
    "BaseExperimentTracker",
    "WandBTracker",
    "TensorBoardTracker",
    "MLflowTracker",
    "TrackerFactory",
    "create_tracker",
    # Utilities
    "get_device_manager",
    "DeviceManager",
    "initialize_weights",
    "WeightInitializer",
    "TensorValidator",
    "ArrayValidator",
    "InputValidator",
    # Optimization
    "GradientClipper",
    "GradientAccumulator",
    "GradientMonitor",
    "LearningRateFinder",
    "LearningRateScheduler",
    "WarmupScheduler",
    "ModelQuantizer",
    "ModelPruner",
    "ModelCompressor",
    # Debugging
    "TrainingDebugger",
    "InferenceDebugger",
    "GradientDebugger",
    "NaNDetector",
    # Monitoring
    "PerformanceMonitor",
    "MemoryMonitor",
    "TrainingMonitor",
    "ModelMonitor",
    # Profiling
    "ModelProfiler",
    "TrainingProfiler",
    "InferenceProfiler",
    # Serialization
    "ModelSerializer",
    "ConfigSerializer",
    "DataSerializer",
    # Logging
    "LoggerFactory",
    "create_logger",
    "TrainingLogger",
    "InferenceLogger",
    "MetricsLogger",
    # Visualization
    "MetricsPlotter",
    "TrainingPlotter",
    "ModelVisualizer",
    "ConfusionMatrixPlotter",
    # Export
    "ModelExporter",
    "ONNXExporter",
    "TorchScriptExporter",
    # Original exports
    "DeepGenreClassifier",
    "DeepMoodDetector",
    "MultiTaskMusicModel",
    "OriginalTransformerEncoder",
    "DeepMusicAnalyzer",
    "get_deep_analyzer",
    # Enhanced components
    "DeviceContext",
    "TrainingContext",
    "EnhancedMixedPrecisionStrategy",
    "EnhancedDataLoader",
    "SmartBatchSampler",
    "GradientAnalyzer",
    "GradientMonitor",
    "EnhancedTransformerWrapper",
    # Interfaces
    "IModel",
    "IEmbeddingModel",
    "IClassifier",
    "ITrainingStrategy",
    "ILossFunction",
    "IOptimizer",
    "IScheduler",
    "IDataTransform",
    "IDataAugmentation",
    "ITrainingCallback",
    "IInferencePipeline",
    "IMonitor",
    "IProfiler",
    "ICheckpointManager",
    "IExperimentTracker",
    "IFactory",
    "IDeviceManager",
    # Specialized factories
    "ModelFactory",
    "TrainingFactory",
    # Training executors
    "BaseTrainingExecutor",
    "StandardTrainingExecutor",
    # Data loaders
    "BaseDataset",
    "BaseDataLoaderFactory",
    "StandardDataLoaderFactory",
    # Attention submodules
    "ScaledDotProductAttentionV2",
    "MultiHeadAttentionV2",
    # Loss submodules
    "ClassificationLossV2",
    "FocalLossV2",
    "LabelSmoothingLossV2",
    "RegressionLossV2",
    # Evaluation metrics submodules
    "ClassificationMetricsV2",
    "RegressionMetricsV2",
    # Inference pipelines
    "BatchInferencePipeline",
    # Builders
    "ModelBuilder",
]

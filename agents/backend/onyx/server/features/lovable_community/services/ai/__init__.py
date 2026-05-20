"""
AI Services for Lovable Community

Modular deep learning system with clear separation of concerns:

Modules:
- core: Base classes and infrastructure
- data: Data loading, preprocessing, augmentation
- training: Training loops, fine-tuning, callbacks
- evaluation: Metrics, evaluators, cross-validation
- optimization: Quantization, pruning, ONNX export
- utils: Debugging, profiling, visualization, multi-GPU
- services: High-level AI services (embeddings, sentiment, etc.)
- tracking: Experiment tracking, model versioning
- interfaces: User interfaces (Gradio)

All modules follow best practices and are production-ready.
"""

# Core infrastructure
try:
    from .core import BaseAIService
except ImportError:
    from .base_service import BaseAIService

# Data processing
try:
    from .data import (
        TextDataset,
        BatchProcessor,
        TextSample,
        preprocess_text,
        batch_texts,
        collate_texts,
        TextPreprocessor,
        TokenizationUtils,
        DataAugmentation,
        FeatureExtractor,
        create_preprocessing_pipeline
    )
except ImportError:
    from .data_loader import (
        TextDataset,
        BatchProcessor,
        TextSample,
        preprocess_text,
        batch_texts,
        collate_texts
    )
    from .preprocessing_utils import (
        TextPreprocessor,
        TokenizationUtils,
        DataAugmentation,
        FeatureExtractor,
        create_preprocessing_pipeline
    )

# Training
try:
    from .training import (
        LoRAFineTuner,
        FullFineTuner,
        EarlyStopping,
        ModelCheckpoint,
        TrainingMetrics,
        Trainer
    )
except ImportError:
    from .fine_tuning import LoRAFineTuner, FullFineTuner
    from .training_utils import (
        EarlyStopping,
        ModelCheckpoint,
        TrainingMetrics,
        Trainer
    )

# Evaluation
try:
    from .evaluation import (
        ClassificationMetrics,
        ModelEvaluator,
        cross_validate
    )
except ImportError:
    from .evaluation_utils import (
        ClassificationMetrics,
        ModelEvaluator,
        cross_validate
    )

# Optimization
try:
    from .optimization import (
        ModelQuantizer,
        ModelPruner,
        ONNXExporter,
        compare_models
    )
except ImportError:
    from .model_optimization import (
        ModelQuantizer,
        ModelPruner,
        ONNXExporter,
        compare_models
    )

# Utilities
try:
    from .utils import (
        NaNInfDetector,
        GradientChecker,
        MemoryProfiler,
        PerformanceProfiler,
        detect_anomaly,
        gradient_checkpointing,
        enable_debug_mode,
        disable_debug_mode,
        MultiGPUTrainer,
        init_distributed,
        cleanup_distributed,
        TrainingVisualizer,
        ModelVisualizer,
        MetricsVisualizer
    )
except ImportError:
    from .debugging_utils import (
        NaNInfDetector,
        GradientChecker,
        MemoryProfiler,
        PerformanceProfiler,
        detect_anomaly,
        gradient_checkpointing,
        enable_debug_mode,
        disable_debug_mode
    )
    from .multi_gpu_utils import (
        MultiGPUTrainer,
        init_distributed,
        cleanup_distributed
    )
    from .visualization_utils import (
        TrainingVisualizer,
        ModelVisualizer,
        MetricsVisualizer
    )

# Services
try:
    from .services import (
        EmbeddingService,
        EmbeddingServiceRefactored,
        SentimentService,
        SentimentServiceRefactored,
        ModerationService,
        ModerationServiceRefactored,
        TextGenerationService,
        DiffusionService,
        RecommendationService
    )
except ImportError:
    from .embedding_service import EmbeddingService
    from .embedding_service_refactored import EmbeddingService as EmbeddingServiceRefactored
    from .sentiment_service import SentimentService
    from .sentiment_service_refactored import SentimentServiceRefactored
    from .moderation_service import ModerationService
    from .moderation_service_refactored import ModerationServiceRefactored
    from .text_generation_service import TextGenerationService
    from .diffusion_service import DiffusionService
    from .recommendation_service import RecommendationService

# Tracking
try:
    from .tracking import (
        ExperimentTracker,
        load_model_config,
        ModelVersion,
        ModelRegistry,
        compare_model_versions
    )
except ImportError:
    from .experiment_tracker import ExperimentTracker, load_model_config
    from .model_versioning import (
        ModelVersion,
        ModelRegistry,
        compare_model_versions
    )

# Interfaces
try:
    from .interfaces import GradioInterface
except ImportError:
    from .gradio_interface import GradioInterface

# Models
try:
    from .models import (
        ClassificationHead,
        RegressionHead,
        TransformerClassifier,
        MultiTaskModel,
        WeightInitializer,
        AttentionVisualizer
    )
except ImportError:
    # Models not available
    pass

# Config
try:
    from .config import (
        ConfigManager,
        get_config_manager,
        ModelConfig,
        TrainingConfig,
        LoRAConfig
    )
except ImportError:
    # Config not available
    pass

# Pipelines
try:
    from .pipelines import TrainingPipeline
except ImportError:
    # Pipelines not available
    pass

# Improved Services (Best Practices)
try:
    from .base_service_improved import BaseAIServiceImproved
    from .embedding_service_improved import EmbeddingServiceImproved
    from .training_improved import ImprovedTrainer, EarlyStopping as EarlyStoppingImproved
except ImportError:
    # Improved services not available
    BaseAIServiceImproved = None
    EmbeddingServiceImproved = None
    ImprovedTrainer = None
    EarlyStoppingImproved = None

# Export all public APIs organized by module
__all__ = [
    # Core
    "BaseAIService",
    
    # Data
    "TextDataset",
    "BatchProcessor",
    "TextSample",
    "preprocess_text",
    "batch_texts",
    "collate_texts",
    "TextPreprocessor",
    "TokenizationUtils",
    "DataAugmentation",
    "FeatureExtractor",
    "create_preprocessing_pipeline",
    
    # Training
    "LoRAFineTuner",
    "FullFineTuner",
    "EarlyStopping",
    "ModelCheckpoint",
    "TrainingMetrics",
    "Trainer",
    
    # Evaluation
    "ClassificationMetrics",
    "ModelEvaluator",
    "cross_validate",
    
    # Optimization
    "ModelQuantizer",
    "ModelPruner",
    "ONNXExporter",
    "compare_models",
    
    # Utils
    "NaNInfDetector",
    "GradientChecker",
    "MemoryProfiler",
    "PerformanceProfiler",
    "detect_anomaly",
    "gradient_checkpointing",
    "enable_debug_mode",
    "disable_debug_mode",
    "MultiGPUTrainer",
    "init_distributed",
    "cleanup_distributed",
    "TrainingVisualizer",
    "ModelVisualizer",
    "MetricsVisualizer",
    
    # Services
    "EmbeddingService",
    "EmbeddingServiceRefactored",
    "SentimentService",
    "SentimentServiceRefactored",
    "ModerationService",
    "ModerationServiceRefactored",
    "TextGenerationService",
    "DiffusionService",
    "RecommendationService",
    
    # Tracking
    "ExperimentTracker",
    "load_model_config",
    "ModelVersion",
    "ModelRegistry",
    "compare_model_versions",
    
    # Interfaces
    "GradioInterface",
    
    # Models
    "ClassificationHead",
    "RegressionHead",
    "TransformerClassifier",
    "MultiTaskModel",
    "WeightInitializer",
    "AttentionVisualizer",
    
    # Config
    "ConfigManager",
    "get_config_manager",
    "ModelConfig",
    "TrainingConfig",
    "LoRAConfig",
    
    # Pipelines
    "TrainingPipeline",
    
    # Improved Services (Best Practices)
    "BaseAIServiceImproved",
    "EmbeddingServiceImproved",
    "ImprovedTrainer",
    "EarlyStopping",
]


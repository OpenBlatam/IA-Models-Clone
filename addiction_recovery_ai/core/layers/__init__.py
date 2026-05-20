"""
Layered Architecture - Ultra Modular Design
Separates concerns into distinct layers with clear interfaces
"""

from typing import Protocol, Optional, Any, Dict, List
import logging

logger = logging.getLogger(__name__)

# Layer 1: Data Layer - Raw data handling and preprocessing
from .data_layer import (
    DataLoader,
    DataProcessor,
    DataValidator,
    DatasetFactory,
    DataPipeline
)

# Layer 2: Model Layer - Model definitions and architectures
from .model_layer import (
    ModelFactory,
    ModelRegistry,
    ModelBuilder,
    ModelConfig,
    ModelLoader
)

# Layer 3: Training Layer - Training loops, optimizers, schedulers
from .training_layer import (
    TrainerFactory,
    TrainingPipeline,
    OptimizerFactory,
    SchedulerFactory,
    TrainingConfig
)

# Layer 4: Inference Layer - Prediction and generation
from .inference_layer import (
    InferenceEngine,
    PredictorFactory,
    InferencePipeline,
    BatchProcessor
)

# Layer 5: Service Layer - High-level business logic
from .service_layer import (
    ServiceFactory,
    ServiceRegistry,
    ServiceContainer,
    ServiceConfig
)

# Layer 6: Interface Layer - API and external interfaces
from .interface_layer import (
    APIHandler,
    InterfaceFactory,
    RequestProcessor,
    ResponseFormatter
)

# Adapters and Integration
from .adapters import (
    ModelAdapter,
    PredictorAdapter,
    ServiceAdapter,
    IntegrationHelper
)

from .integration import (
    WorkflowBuilder,
    CompleteWorkflow,
    create_sentiment_workflow,
    create_training_workflow,
    create_inference_workflow
)

# Utilities
from .utils import (
    quick_model,
    quick_inference_engine,
    quick_data_pipeline,
    quick_service_container,
    quick_workflow,
    get_optimal_device,
    setup_mixed_precision,
    merge_configs,
    validate_config
)

# Micro-Modules - Ultra-Granular Components
from .micro_modules import (
    # Data Processors
    Normalizer,
    StandardNormalizer,
    MinMaxNormalizer,
    Tokenizer,
    SimpleTokenizer,
    HuggingFaceTokenizer,
    Padder,
    ZeroPadder,
    RepeatPadder,
    Augmenter,
    NoiseAugmenter,
    DropoutAugmenter,
    Validator,
    TensorValidator,
    ShapeValidator,
    RangeValidator,
    # Model Components
    ModelInitializer,
    ModelCompiler,
    ModelOptimizer,
    ModelQuantizer,
    # Training Components
    LossCalculator,
    GradientManager,
    LearningRateManager,
    CheckpointManager,
    # Inference Components
    BatchProcessor,
    CacheManager,
    OutputFormatter,
    PostProcessor,
)

__all__ = [
    # Data Layer
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "DatasetFactory",
    "DataPipeline",
    # Model Layer
    "ModelFactory",
    "ModelRegistry",
    "ModelBuilder",
    "ModelConfig",
    "ModelLoader",
    # Training Layer
    "TrainerFactory",
    "TrainingPipeline",
    "OptimizerFactory",
    "SchedulerFactory",
    "TrainingConfig",
    # Inference Layer
    "InferenceEngine",
    "PredictorFactory",
    "InferencePipeline",
    "BatchProcessor",
    # Service Layer
    "ServiceFactory",
    "ServiceRegistry",
    "ServiceContainer",
    "ServiceConfig",
    # Interface Layer
    "APIHandler",
    "InterfaceFactory",
    "RequestProcessor",
    "ResponseFormatter",
    # Adapters and Integration
    "ModelAdapter",
    "PredictorAdapter",
    "ServiceAdapter",
    "IntegrationHelper",
    "WorkflowBuilder",
    "CompleteWorkflow",
    "create_sentiment_workflow",
    "create_training_workflow",
    "create_inference_workflow",
    # Utilities
    "quick_model",
    "quick_inference_engine",
    "quick_data_pipeline",
    "quick_service_container",
    "quick_workflow",
    "get_optimal_device",
    "setup_mixed_precision",
    "merge_configs",
    "validate_config",
    # Micro-Modules
    "Normalizer",
    "StandardNormalizer",
    "MinMaxNormalizer",
    "Tokenizer",
    "SimpleTokenizer",
    "HuggingFaceTokenizer",
    "Padder",
    "ZeroPadder",
    "RepeatPadder",
    "Augmenter",
    "NoiseAugmenter",
    "DropoutAugmenter",
    "Validator",
    "TensorValidator",
    "ShapeValidator",
    "RangeValidator",
    "ModelInitializer",
    "ModelCompiler",
    "ModelOptimizer",
    "ModelQuantizer",
    "LossCalculator",
    "GradientManager",
    "LearningRateManager",
    "CheckpointManager",
    "BatchProcessor",
    "CacheManager",
    "OutputFormatter",
    "PostProcessor",
]


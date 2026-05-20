"""
Core logic para análisis musical
"""

from .models import (
    DeepGenreClassifier,
    DeepMoodDetector,
    MultiTaskMusicModel,
    TransformerMusicEncoder,
    DeepMusicAnalyzer,
    get_deep_analyzer
)

from .transformers import (
    AttentionVisualizer,
    TransformerFineTuner,
    MusicTransformerEncoder
)

from .composition import (
    ModelComposer,
    ComposedModel,
    SequentialComposer,
    ParallelComposer,
    ParallelModel
)

from .processing import (
    ProcessingStage,
    ProcessingResult,
    ProcessingLayer,
    PreprocessingLayer,
    FeatureExtractionLayer,
    MLInferenceLayer,
    PostprocessingLayer,
    ValidationLayer,
    ProcessingPipeline,
    create_default_pipeline
)

from .ml_audio import (
    AudioFeatures,
    MLPrediction,
    AudioFeatureExtractor,
    GenreClassifier,
    MLMusicAnalyzer,
    get_ml_analyzer
)

from .events import (
    Event,
    EventBus,
    get_event_bus,
    subscribe,
    publish
)

from .di import (
    DIContainer,
    get_container,
    register_service,
    get_service
)

from .registry import (
    ComponentRegistry,
    get_registry,
    register_model,
    register_loss,
    register_optimizer,
    register_scheduler
)

from .model_manager import ModelManager

__all__ = [
    # Models
    "DeepGenreClassifier",
    "DeepMoodDetector",
    "MultiTaskMusicModel",
    "TransformerMusicEncoder",
    "DeepMusicAnalyzer",
    "get_deep_analyzer",
    # Transformers
    "AttentionVisualizer",
    "TransformerFineTuner",
    "MusicTransformerEncoder",
    # Composition
    "ModelComposer",
    "ComposedModel",
    "SequentialComposer",
    "ParallelComposer",
    "ParallelModel",
    # Processing
    "ProcessingStage",
    "ProcessingResult",
    "ProcessingLayer",
    "PreprocessingLayer",
    "FeatureExtractionLayer",
    "MLInferenceLayer",
    "PostprocessingLayer",
    "ValidationLayer",
    "ProcessingPipeline",
    "create_default_pipeline",
    # ML Audio
    "AudioFeatures",
    "MLPrediction",
    "AudioFeatureExtractor",
    "GenreClassifier",
    "MLMusicAnalyzer",
    "get_ml_analyzer",
    # Events
    "Event",
    "EventBus",
    "get_event_bus",
    "subscribe",
    "publish",
    # Dependency Injection
    "DIContainer",
    "get_container",
    "register_service",
    "get_service",
    # Registry
    "ComponentRegistry",
    "get_registry",
    "register_model",
    "register_loss",
    "register_optimizer",
    "register_scheduler",
    # Model Manager
    "ModelManager",
]


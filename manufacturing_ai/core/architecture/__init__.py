"""
Architecture Module
===================

Módulo de arquitecturas avanzadas y componentes reutilizables.
"""

from .attention_layers import MultiHeadAttention, SelfAttention, CrossAttention
from .residual_blocks import ResidualBlock, ResidualConnection
from .normalization import LayerNorm, BatchNorm1d, GroupNorm
from .activations import GELU, Swish, Mish
from .model_builder import ModelBuilder, ArchitectureConfig
from .component_factory import ComponentFactory
from .distributed_training import DistributedTrainingManager, get_distributed_training_manager
from .advanced_models import AdvancedQualityPredictor, AdvancedProcessOptimizer
from .advanced_training import (
    MixedPrecisionTrainer,
    GradientAccumulator,
    AdvancedTrainer
)
from .profiling import ModelProfiler, ModelOptimizer
from .checkpointing import CheckpointManager
from .embeddings import (
    PositionalEncoding,
    LearnablePositionalEncoding,
    TokenEmbedding,
    FeatureEmbedding
)
from .ensembling import (
    ModelEnsemble,
    StackingEnsemble,
    EnsembleManager,
    get_ensemble_manager
)
from .data_augmentation import (
    AdvancedImageAugmentation,
    FeatureAugmentation,
    MixUp,
    CutMix
)
from .performance_optimization import (
    ModelCache,
    FastInference,
    MemoryOptimizer,
    BatchProcessor,
    AsyncInference,
    get_fast_inference,
    get_memory_optimizer,
    get_batch_processor
)
from .quantization import (
    ModelQuantizer,
    FP16Inference
)
from .speed_optimizations import (
    ModelWarmup,
    ParallelInference,
    TensorOptimizer,
    DataLoaderOptimizer,
    InferencePipeline,
    get_tensor_optimizer,
    get_dataloader_optimizer
)
from .data_pipelines import (
    DataPipeline,
    ParallelDataPipeline,
    StreamingPipeline,
    AsyncDataPipeline,
    PrefetchDataLoader,
    BatchAggregator,
    DataTransformer,
    PipelineBuilder,
    create_data_pipeline,
    create_parallel_pipeline,
    create_streaming_pipeline,
    create_async_pipeline
)
from .realtime_streaming import (
    StreamType,
    StreamEvent,
    StreamManager,
    WebSocketStreamHandler,
    ProductionStreamer,
    QualityStreamer,
    MonitoringStreamer,
    OptimizationStreamer,
    StreamAggregator,
    get_stream_manager
)
from .model_serving import (
    ServingMode,
    ServingRequest,
    ServingResponse,
    ModelServer,
    ModelServerRegistry,
    LoadBalancer,
    get_model_server_registry
)
from .smart_cache import (
    SmartCache,
    cached,
    PredictionCache,
    MultiLevelCache,
    get_prediction_cache
)

__all__ = [
    # Core components
    "MultiHeadAttention",
    "SelfAttention",
    "CrossAttention",
    "ResidualBlock",
    "ResidualConnection",
    "LayerNorm",
    "BatchNorm1d",
    "GroupNorm",
    "GELU",
    "Swish",
    "Mish",
    "ModelBuilder",
    "ArchitectureConfig",
    "ComponentFactory",
    # Distributed training
    "DistributedTrainingManager",
    "get_distributed_training_manager",
    # Advanced models
    "AdvancedQualityPredictor",
    "AdvancedProcessOptimizer",
    # Advanced training
    "MixedPrecisionTrainer",
    "GradientAccumulator",
    "AdvancedTrainer",
    # Profiling
    "ModelProfiler",
    "ModelOptimizer",
    # Checkpointing
    "CheckpointManager",
    # Embeddings
    "PositionalEncoding",
    "LearnablePositionalEncoding",
    "TokenEmbedding",
    "FeatureEmbedding",
    # Ensembling
    "ModelEnsemble",
    "StackingEnsemble",
    "EnsembleManager",
    "get_ensemble_manager",
    # Data augmentation
    "AdvancedImageAugmentation",
    "FeatureAugmentation",
    "MixUp",
    "CutMix",
    # Performance optimization
    "ModelCache",
    "FastInference",
    "MemoryOptimizer",
    "BatchProcessor",
    "AsyncInference",
    "get_fast_inference",
    "get_memory_optimizer",
    "get_batch_processor",
    # Quantization
    "ModelQuantizer",
    "FP16Inference",
    # Speed optimizations
    "ModelWarmup",
    "ParallelInference",
    "TensorOptimizer",
    "DataLoaderOptimizer",
    "InferencePipeline",
    "get_tensor_optimizer",
    "get_dataloader_optimizer",
    # Data pipelines
    "DataPipeline",
    "ParallelDataPipeline",
    "StreamingPipeline",
    "AsyncDataPipeline",
    "PrefetchDataLoader",
    "BatchAggregator",
    "DataTransformer",
    "PipelineBuilder",
    "create_data_pipeline",
    "create_parallel_pipeline",
    "create_streaming_pipeline",
    "create_async_pipeline",
    # Real-time streaming
    "StreamType",
    "StreamEvent",
    "StreamManager",
    "WebSocketStreamHandler",
    "ProductionStreamer",
    "QualityStreamer",
    "MonitoringStreamer",
    "OptimizationStreamer",
    "StreamAggregator",
    "get_stream_manager",
    # Model serving
    "ServingMode",
    "ServingRequest",
    "ServingResponse",
    "ModelServer",
    "ModelServerRegistry",
    "LoadBalancer",
    "get_model_server_registry",
    # Smart cache
    "SmartCache",
    "cached",
    "PredictionCache",
    "MultiLevelCache",
    "get_prediction_cache",
]


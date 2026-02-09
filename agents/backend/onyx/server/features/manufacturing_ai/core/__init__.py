"""
Manufacturing AI Core Module
=============================

Módulo principal para sistemas de manufactura inteligente.
"""

from .production_planner import ProductionPlanner, ProductionOrder, OrderStatus
from .quality_control import QualityController, QualityCheck, QualityResult
from .process_optimizer import ProcessOptimizer, ManufacturingProcess, OptimizationResult
from .monitoring import ManufacturingMonitor, ProductionMetrics, EquipmentStatus
from .demand_forecasting import (
    DemandForecastingSystem,
    DemandForecast,
    get_demand_forecasting_system
)
from .predictive_maintenance import (
    PredictiveMaintenanceSystem,
    MaintenancePrediction,
    SensorData,
    get_predictive_maintenance_system
)
from .capacity_optimizer import (
    CapacityOptimizer,
    CapacityPlan,
    ResourceAllocation,
    get_capacity_optimizer
)
from .inventory_optimizer import (
    IntelligentInventorySystem,
    InventoryItem,
    InventoryPrediction,
    InventoryStatus,
    get_intelligent_inventory_system
)
from .production_route_optimizer import (
    ProductionRouteOptimizer,
    ProductionStep,
    ProductionRoute,
    get_production_route_optimizer
)
from .production_analyzer import (
    AdvancedProductionAnalyzer,
    ProductionAnalysis,
    get_advanced_production_analyzer
)

# Architecture components
from .architecture import (
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    ResidualBlock,
    ResidualConnection,
    LayerNorm,
    BatchNorm1d,
    GroupNorm,
    GELU,
    Swish,
    Mish,
    ModelBuilder,
    ArchitectureConfig,
    ComponentFactory,
    DistributedTrainingManager,
    get_distributed_training_manager,
    AdvancedQualityPredictor,
    AdvancedProcessOptimizer,
    MixedPrecisionTrainer,
    GradientAccumulator,
    AdvancedTrainer,
    ModelProfiler,
    ModelOptimizer,
    CheckpointManager,
    PositionalEncoding,
    LearnablePositionalEncoding,
    TokenEmbedding,
    FeatureEmbedding,
    ModelEnsemble,
    StackingEnsemble,
    EnsembleManager,
    get_ensemble_manager,
    AdvancedImageAugmentation,
    FeatureAugmentation,
    MixUp,
    CutMix,
    ModelCache,
    FastInference,
    MemoryOptimizer,
    BatchProcessor,
    AsyncInference,
    get_fast_inference,
    get_memory_optimizer,
    get_batch_processor,
    ModelQuantizer,
    FP16Inference
)

__all__ = [
    "ProductionPlanner",
    "ProductionOrder",
    "OrderStatus",
    "QualityController",
    "QualityCheck",
    "QualityResult",
    "ProcessOptimizer",
    "ManufacturingProcess",
    "OptimizationResult",
    "ManufacturingMonitor",
    "ProductionMetrics",
    "EquipmentStatus",
    "DemandForecastingSystem",
    "DemandForecast",
    "get_demand_forecasting_system",
    "PredictiveMaintenanceSystem",
    "MaintenancePrediction",
    "SensorData",
    "get_predictive_maintenance_system",
    "CapacityOptimizer",
    "CapacityPlan",
    "ResourceAllocation",
    "get_capacity_optimizer",
    "IntelligentInventorySystem",
    "InventoryItem",
    "InventoryPrediction",
    "InventoryStatus",
    "get_intelligent_inventory_system",
    "ProductionRouteOptimizer",
    "ProductionStep",
    "ProductionRoute",
    "get_production_route_optimizer",
    "AdvancedProductionAnalyzer",
    "ProductionAnalysis",
    "get_advanced_production_analyzer",
    # Architecture
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
    "DistributedTrainingManager",
    "get_distributed_training_manager",
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
]


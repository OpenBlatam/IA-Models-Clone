"""
Core Module
Fundamental interfaces, factories, and builders.
"""

# Interfaces
from .interfaces import (
    IModelLoader,
    IInferenceEngine,
    ITrainer,
    IEvaluator,
    IDataProcessor,
    IOptimizer,
    IQuantizer,
    IProfiler,
    IRegistry,
    IConfigManager,
    ILogger,
)

# Factories
from .factories import (
    ModelLoaderFactory,
    OptimizerFactory,
    LossFunctionFactory,
    DeviceFactory,
    ComponentFactory,
)

# Builders
from .builders import (
    TrainingPipelineBuilder,
    InferencePipelineBuilder,
    ModelOptimizationBuilder,
)

# Loss functions
from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    DiceLoss,
    ContrastiveLoss,
)

__all__ = [
    # Interfaces
    "IModelLoader",
    "IInferenceEngine",
    "ITrainer",
    "IEvaluator",
    "IDataProcessor",
    "IOptimizer",
    "IQuantizer",
    "IProfiler",
    "IRegistry",
    "IConfigManager",
    "ILogger",
    # Factories
    "ModelLoaderFactory",
    "OptimizerFactory",
    "LossFunctionFactory",
    "DeviceFactory",
    "ComponentFactory",
    # Builders
    "TrainingPipelineBuilder",
    "InferencePipelineBuilder",
    "ModelOptimizationBuilder",
    # Loss functions
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "ContrastiveLoss",
]




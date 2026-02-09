"""
Training Micro-Modules
Organized by category for maximum modularity
"""

from ..losses import (
    LossBase,
    MSELoss,
    MAELoss,
    BCELoss,
    CrossEntropyLoss,
    SmoothL1Loss,
    FocalLoss,
    LossFactory
)

from .gradient_manager import (
    GradientManagerBase,
    GradientClipper,
    GradientChecker,
    GradientAccumulator,
    GradientNormalizer,
    GradientManagerFactory
)

from .lr_manager import (
    LRManagerBase,
    StepLRManager,
    ExponentialLRManager,
    CosineAnnealingLRManager,
    ReduceLROnPlateauManager,
    OneCycleLRManager,
    WarmupLRManager,
    LRManagerFactory
)

from .checkpoint_manager import (
    CheckpointManagerBase,
    FullCheckpointManager,
    StateDictCheckpointManager,
    BestModelCheckpointManager,
    PeriodicCheckpointManager,
    CheckpointManagerFactory
)

# Backward compatibility
from ..training_components import (
    LossCalculator,
    GradientManager,
    LearningRateManager,
    CheckpointManager
)

__all__ = [
    # Loss Functions
    "LossBase",
    "MSELoss",
    "MAELoss",
    "BCELoss",
    "CrossEntropyLoss",
    "SmoothL1Loss",
    "FocalLoss",
    "LossFactory",
    # Gradient Management
    "GradientManagerBase",
    "GradientClipper",
    "GradientChecker",
    "GradientAccumulator",
    "GradientNormalizer",
    "GradientManagerFactory",
    # Learning Rate Management
    "LRManagerBase",
    "StepLRManager",
    "ExponentialLRManager",
    "CosineAnnealingLRManager",
    "ReduceLROnPlateauManager",
    "OneCycleLRManager",
    "WarmupLRManager",
    "LRManagerFactory",
    # Checkpoint Management
    "CheckpointManagerBase",
    "FullCheckpointManager",
    "StateDictCheckpointManager",
    "BestModelCheckpointManager",
    "PeriodicCheckpointManager",
    "CheckpointManagerFactory",
    # Backward Compatibility
    "LossCalculator",
    "GradientManager",
    "LearningRateManager",
    "CheckpointManager",
]


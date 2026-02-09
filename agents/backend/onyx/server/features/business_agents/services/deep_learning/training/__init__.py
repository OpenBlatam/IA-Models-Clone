"""Training utilities for deep learning service."""

from .trainer import TrainingManager
from .early_stopping import EarlyStopping
from .checkpoint import CheckpointManager
from .lora import apply_lora, LoraConfig
from .losses import (
    FocalLoss, LabelSmoothingLoss, DiceLoss, CombinedLoss, get_loss_function
)
from .callbacks import (
    Callback, CallbackList, LearningRateSchedulerCallback,
    ModelCheckpointCallback, EarlyStoppingCallback,
    TensorBoardCallback, WandBCallback
)

# Optional optimized trainer
try:
    from .optimized_trainer import OptimizedTrainingManager
    OPTIMIZED_TRAINER_AVAILABLE = True
except ImportError:
    OPTIMIZED_TRAINER_AVAILABLE = False
    OptimizedTrainingManager = None

__all__ = [
    "TrainingManager",
    "EarlyStopping",
    "CheckpointManager",
    "apply_lora",
    "LoraConfig",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "CombinedLoss",
    "get_loss_function",
    "Callback",
    "CallbackList",
    "LearningRateSchedulerCallback",
    "ModelCheckpointCallback",
    "EarlyStoppingCallback",
    "TensorBoardCallback",
    "WandBCallback",
]

if OPTIMIZED_TRAINER_AVAILABLE:
    __all__.append("OptimizedTrainingManager")


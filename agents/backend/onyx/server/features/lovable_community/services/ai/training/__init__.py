"""
Training Module

Handles all training-related operations organized into sub-modules:
- fine_tuning: LoRA and full fine-tuning
- trainer: Training pipeline and metrics
- callbacks: Training callbacks (early stopping, checkpointing)
"""

from .fine_tuning import (
    LoRAFineTuner,
    FullFineTuner
)

from .trainer import (
    Trainer,
    TrainingMetrics
)

from .callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

try:
    from .advanced_trainer import AdvancedTrainer
except ImportError:
    AdvancedTrainer = None

__all__ = [
    "LoRAFineTuner",
    "FullFineTuner",
    "Trainer",
    "TrainingMetrics",
    "EarlyStopping",
    "ModelCheckpoint",
    "AdvancedTrainer",
]


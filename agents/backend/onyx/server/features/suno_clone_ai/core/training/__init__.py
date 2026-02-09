"""
Training Module

Provides:
- Enhanced training pipeline with best practices
- Evaluation metrics
- Cross-validation support
- Optimizers and schedulers
- Loss functions
- Training callbacks
- Experiment tracking integration
"""

from .enhanced_training import (
    EnhancedTrainingPipeline,
    EvaluationMetrics,
    create_train_val_test_split,
    cross_validate
)

from .optimizers import (
    create_optimizer,
    create_scheduler,
    create_warmup_scheduler,
    get_parameter_groups
)

from .losses import (
    MSELoss,
    MAELoss,
    SpectralLoss,
    CombinedLoss,
    create_loss_function
)

from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)

__all__ = [
    # Training pipeline
    "EnhancedTrainingPipeline",
    "EvaluationMetrics",
    "create_train_val_test_split",
    "cross_validate",
    # Optimizers
    "create_optimizer",
    "create_scheduler",
    "create_warmup_scheduler",
    "get_parameter_groups",
    # Losses
    "MSELoss",
    "MAELoss",
    "SpectralLoss",
    "CombinedLoss",
    "create_loss_function",
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateMonitor"
]


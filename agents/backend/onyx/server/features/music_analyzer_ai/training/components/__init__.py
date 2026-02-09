"""
Modular Training Components
Separated into individual modules for better modularity
"""

# Import losses (backward compatibility)
try:
    from .losses import (
        ClassificationLoss,
        RegressionLoss,
        MultiTaskLoss,
        FocalLoss,
        LabelSmoothingLoss
    )
except ImportError:
    # Fallback to submodule
    from .losses.classification import (
        ClassificationLoss,
        FocalLoss,
        LabelSmoothingLoss
    )
    from .losses.regression import RegressionLoss
    # MultiTaskLoss might need to be imported separately
    try:
        from .losses import MultiTaskLoss
    except ImportError:
        MultiTaskLoss = None
# Import from submodules
from .optimizers import (
    OptimizerFactory,
    create_optimizer
)
from .schedulers import (
    SchedulerFactory,
    create_scheduler,
    WarmupScheduler
)
from .callbacks import (
    TrainingCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LearningRateCallback,
    MetricsCallback
)

__all__ = [
    # Losses
    "ClassificationLoss",
    "RegressionLoss",
    "MultiTaskLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    # Optimizers
    "OptimizerFactory",
    "create_optimizer",
    # Schedulers
    "SchedulerFactory",
    "create_scheduler",
    "WarmupScheduler",
    # Callbacks
    "TrainingCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LearningRateCallback",
    "MetricsCallback",
]


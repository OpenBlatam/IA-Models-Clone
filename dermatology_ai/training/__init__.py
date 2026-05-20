"""
Training module
Separated from core for better organization
Enhanced with distributed training, profiling, and callbacks
"""

from .trainer import Trainer, create_data_loaders
from .losses import MultiTaskLoss, ConditionLoss, MetricLoss, FocalLoss, DiceLoss
from .optimizers import get_optimizer, get_scheduler, get_optimizer_and_scheduler
from .metrics import MetricCalculator, ClassificationMetrics, RegressionMetrics
from .distributed import (
    setup_distributed,
    wrap_model_for_distributed,
    get_world_size,
    get_rank,
    is_main_process,
    reduce_tensor,
    synchronize,
    DistributedSampler
)

# Try to import refactored components (may not exist yet)
try:
    from ml.training.trainer_refactored import RefactoredTrainer
    from ml.training.pipeline import TrainingPipeline
    from ml.training.callbacks import (
        TrainingCallback,
        EarlyStoppingCallback,
        ModelCheckpointCallback,
        LearningRateSchedulerCallback,
        MetricsLoggingCallback
    )
    REFACTORED_AVAILABLE = True
except ImportError:
    REFACTORED_AVAILABLE = False

__all__ = [
    # Trainer
    'Trainer',
    'create_data_loaders',
    # Losses
    'MultiTaskLoss',
    'ConditionLoss',
    'MetricLoss',
    'FocalLoss',
    'DiceLoss',
    # Optimizers
    'get_optimizer',
    'get_scheduler',
    'get_optimizer_and_scheduler',
    # Metrics
    'MetricCalculator',
    'ClassificationMetrics',
    'RegressionMetrics',
    # Distributed
    'setup_distributed',
    'wrap_model_for_distributed',
    'get_world_size',
    'get_rank',
    'is_main_process',
    'reduce_tensor',
    'synchronize',
    'DistributedSampler'
]

# Add refactored components if available
if REFACTORED_AVAILABLE:
    __all__.extend([
        'RefactoredTrainer',
        'TrainingPipeline',
        'TrainingCallback',
        'EarlyStoppingCallback',
        'ModelCheckpointCallback',
        'LearningRateSchedulerCallback',
        'MetricsLoggingCallback'
    ])


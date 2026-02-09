"""
Training Module - Training Loops and Optimization
=================================================

This module provides training functionality following best practices:
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Multi-GPU support
- Experiment tracking
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

from .trainer import Trainer, TrainingConfig, EarlyStopping
from .optimizers import create_optimizer, create_scheduler

# Try to import advanced optimizers
try:
    from .advanced_optimizers import (
        LearningRateFinder,
        create_optimizer_with_warmup
    )
    ADVANCED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZERS_AVAILABLE = False
    LearningRateFinder = None
    create_optimizer_with_warmup = None

# Try to import distributed training
try:
    from .distributed_training import (
        setup_distributed,
        wrap_model_for_distributed,
        cleanup_distributed,
        DistributedSamplerWrapper
    )
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    setup_distributed = None
    wrap_model_for_distributed = None
    cleanup_distributed = None
    DistributedSamplerWrapper = None

# Try to import callbacks
try:
    from .callbacks import (
        Callback,
        ModelCheckpoint,
        LearningRateScheduler,
        MetricsLogger,
        CallbackList
    )
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False
    Callback = None
    ModelCheckpoint = None
    LearningRateScheduler = None
    MetricsLogger = None
    CallbackList = None

__all__ = [
    "Trainer",
    "TrainingConfig",
    "create_optimizer",
    "create_scheduler",
    "EarlyStopping",
]

if CALLBACKS_AVAILABLE:
    __all__.extend([
        "Callback",
        "ModelCheckpoint",
        "LearningRateScheduler",
        "MetricsLogger",
        "CallbackList"
    ])

if DISTRIBUTED_AVAILABLE:
    __all__.extend([
        "setup_distributed",
        "wrap_model_for_distributed",
        "cleanup_distributed",
        "DistributedSamplerWrapper"
    ])

if ADVANCED_OPTIMIZERS_AVAILABLE:
    __all__.extend([
        "LearningRateFinder",
        "create_optimizer_with_warmup"
    ])

# Try to import advanced schedulers
try:
    from .advanced_schedulers import (
        WarmupScheduler,
        create_warmup_scheduler,
        create_onecycle_scheduler
    )
    ADVANCED_SCHEDULERS_AVAILABLE = True
except ImportError:
    ADVANCED_SCHEDULERS_AVAILABLE = False
    WarmupScheduler = None
    create_warmup_scheduler = None
    create_onecycle_scheduler = None

if ADVANCED_SCHEDULERS_AVAILABLE:
    __all__.extend([
        "WarmupScheduler",
        "create_warmup_scheduler",
        "create_onecycle_scheduler"
    ])


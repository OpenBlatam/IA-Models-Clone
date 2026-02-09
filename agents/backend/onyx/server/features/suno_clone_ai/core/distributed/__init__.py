"""
Distributed Training Module

Provides:
- Multi-GPU training utilities
- Distributed data parallel
- Gradient synchronization
- Distributed training setup
"""

from .distributed_trainer import (
    DistributedTrainer,
    setup_distributed,
    get_distributed_config
)

from .gradient_sync import (
    GradientSynchronizer,
    sync_gradients,
    all_reduce_gradients
)

__all__ = [
    # Distributed training
    "DistributedTrainer",
    "setup_distributed",
    "get_distributed_config",
    # Gradient sync
    "GradientSynchronizer",
    "sync_gradients",
    "all_reduce_gradients"
]




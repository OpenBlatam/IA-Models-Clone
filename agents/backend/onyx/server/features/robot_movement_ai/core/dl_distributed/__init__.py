"""
Distributed Training Module
===========================

Módulo de entrenamiento distribuido.
"""

from .distributed_trainer import (
    DistributedTrainer,
    setup_distributed,
    cleanup_distributed
)

__all__ = [
    'DistributedTrainer',
    'setup_distributed',
    'cleanup_distributed'
]









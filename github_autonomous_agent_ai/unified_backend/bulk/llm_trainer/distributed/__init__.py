"""
Distributed Training Module
===========================

Utilities for distributed and multi-GPU training.

Author: BUL System
Date: 2024
"""

from .distributed_utils import setup_distributed_training, is_distributed, get_world_size, get_rank

__all__ = [
    "setup_distributed_training",
    "is_distributed",
    "get_world_size",
    "get_rank",
]


"""
TruthGPT Distributed Training Module
==================================

TensorFlow-like distributed training utilities for TruthGPT.
"""

from .ddp import DistributedDataParallel
from .horovod import HorovodDistributed
from .ray import RayDistributed
from .base import BaseDistributed

__all__ = [
    'BaseDistributed', 'DistributedDataParallel', 
    'HorovodDistributed', 'RayDistributed'
]



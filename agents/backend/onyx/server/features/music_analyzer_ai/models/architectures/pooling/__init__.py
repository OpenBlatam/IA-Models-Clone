"""
Pooling Submodule
Aggregates various pooling components.
"""

from .mean import MeanPooling
from .max import MaxPooling
from .attention import AttentionPooling
from .adaptive import AdaptivePooling
from .factory import PoolingFactory, create_pooling

__all__ = [
    "MeanPooling",
    "MaxPooling",
    "AttentionPooling",
    "AdaptivePooling",
    "PoolingFactory",
    "create_pooling",
]




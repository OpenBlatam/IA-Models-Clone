"""
Backward compatibility re-export for cluster.py

This file is deprecated. Use infrastructure.clustering.cluster instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.clustering.cluster instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.clustering.cluster import *

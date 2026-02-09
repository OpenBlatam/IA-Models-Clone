"""
Backward compatibility re-export for pattern_learner.py

This file is deprecated. Use ai.pattern_learner instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use ai.pattern_learner instead.",
    DeprecationWarning,
    stacklevel=2
)

from .ai.pattern_learner import *

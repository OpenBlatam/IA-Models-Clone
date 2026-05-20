"""
Backward compatibility re-export for ai_processor.py

This file is deprecated. Use ai.ai_processor instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use ai.ai_processor instead.",
    DeprecationWarning,
    stacklevel=2
)

from .ai.ai_processor import *

"""
Backward compatibility re-export for llm_pipeline.py

This file is deprecated. Use ai.llm_pipeline instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use ai.llm_pipeline instead.",
    DeprecationWarning,
    stacklevel=2
)

from .ai.llm_pipeline import *

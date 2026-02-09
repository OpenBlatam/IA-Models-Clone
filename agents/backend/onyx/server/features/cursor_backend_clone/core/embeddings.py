"""
Backward compatibility re-export for embeddings.py

This file is deprecated. Use ai.embeddings instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use ai.embeddings instead.",
    DeprecationWarning,
    stacklevel=2
)

from .ai.embeddings import *

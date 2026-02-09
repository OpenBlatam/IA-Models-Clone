"""
Model Loader - DEPRECATED

This module is deprecated. Please use the modular version:

    from core.model_loader import ModelLoader, ModelConfig, BackendType

The modular version is located in:
    core/model_loader/

This file is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "core.model_loader (monolithic) is deprecated. "
    "Use core.model_loader (modular) instead. "
    "This file will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from modular version
from .model_loader import (
    ModelType,
    QuantizationType,
    BackendType,
    ModelConfig,
    GenerationConfig,
    ModelLoader,
    create_backend,
    auto_select_backend,
    get_available_backends,
)

__all__ = [
    "ModelType",
    "QuantizationType",
    "BackendType",
    "ModelConfig",
    "GenerationConfig",
    "ModelLoader",
    "create_backend",
    "auto_select_backend",
    "get_available_backends",
]

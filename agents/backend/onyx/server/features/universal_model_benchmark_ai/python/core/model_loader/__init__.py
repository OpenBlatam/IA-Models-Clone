"""
Model Loader Module - Modular model loading system.

This module provides a complete model loading system with:
- Multiple backend support (vLLM, Transformers, llama.cpp)
- Type-safe configuration
- Factory pattern for backend creation
- High-level ModelLoader class

Structure:
- types: Enums and configuration classes
- factory: Backend creation and selection
- loader: Main ModelLoader class
"""

from .types import (
    ModelType,
    QuantizationType,
    BackendType,
    ModelConfig,
    GenerationConfig,
)

from .factory import (
    create_backend,
    auto_select_backend,
    get_available_backends,
    check_vllm_available,
    check_llama_cpp_available,
    check_tensorrt_llm_available,
)

from .loader import ModelLoader


__all__ = [
    # Types
    "ModelType",
    "QuantizationType",
    "BackendType",
    "ModelConfig",
    "GenerationConfig",
    # Factory
    "create_backend",
    "auto_select_backend",
    "get_available_backends",
    "check_vllm_available",
    "check_llama_cpp_available",
    "check_tensorrt_llm_available",
    # Loader
    "ModelLoader",
]













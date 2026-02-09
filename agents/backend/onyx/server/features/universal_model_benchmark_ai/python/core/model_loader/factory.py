"""
Model Loader Factory - Backend creation and selection.

This module provides factory functions for creating backend instances
and auto-selecting the best available backend.
"""

import logging
from typing import Optional

from .types import BackendType
from ..backends.base import BaseBackend

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# BACKEND AVAILABILITY CHECKING
# ════════════════════════════════════════════════════════════════════════════════

def check_vllm_available() -> bool:
    """Check if vLLM is available."""
    try:
        from vllm import LLM
        return True
    except ImportError:
        return False


def check_llama_cpp_available() -> bool:
    """Check if llama.cpp is available."""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


def check_tensorrt_llm_available() -> bool:
    """Check if TensorRT-LLM is available."""
    try:
        import tensorrt_llm
        return True
    except ImportError:
        return False


# ════════════════════════════════════════════════════════════════════════════════
# BACKEND FACTORY
# ════════════════════════════════════════════════════════════════════════════════

def create_backend(backend_type: BackendType) -> BaseBackend:
    """
    Create backend instance.
    
    Args:
        backend_type: Type of backend to create
    
    Returns:
        Backend instance
    
    Raises:
        ValueError: If backend type is invalid
        RuntimeError: If backend is not available
    """
    if backend_type == BackendType.VLLM:
        if not check_vllm_available():
            raise RuntimeError(
                "vLLM is not available. Install with: pip install vllm>=0.2.0"
            )
        from ..backends.vllm_backend import VLLMBackend
        return VLLMBackend()
    
    elif backend_type == BackendType.TRANSFORMERS:
        from ..backends.transformers_backend import TransformersBackend
        return TransformersBackend()
    
    elif backend_type == BackendType.LLAMA_CPP:
        if not check_llama_cpp_available():
            raise RuntimeError(
                "llama.cpp is not available. Install with: pip install llama-cpp-python"
            )
        from ..backends.llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend()
    
    elif backend_type == BackendType.TENSORRT_LLM:
        if not check_tensorrt_llm_available():
            raise RuntimeError(
                "TensorRT-LLM is not available. Install from NVIDIA's repository."
            )
        raise NotImplementedError("TensorRT-LLM backend not yet implemented")
    
    elif backend_type == BackendType.AUTO:
        return auto_select_backend()
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def auto_select_backend() -> BaseBackend:
    """
    Auto-select best available backend.
    
    Priority:
    1. vLLM (fastest, best for GPU)
    2. llama.cpp (good for CPU)
    3. Transformers (fallback, always available)
    
    Returns:
        Best available backend instance
    """
    if check_vllm_available():
        logger.info("Auto-selected vLLM backend")
        from ..backends.vllm_backend import VLLMBackend
        return VLLMBackend()
    
    elif check_llama_cpp_available():
        logger.info("Auto-selected llama.cpp backend")
        from ..backends.llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend()
    
    else:
        logger.info("Auto-selected Transformers backend (fallback)")
        from ..backends.transformers_backend import TransformersBackend
        return TransformersBackend()


def get_available_backends() -> list[BackendType]:
    """
    Get list of available backends.
    
    Returns:
        List of available backend types
    """
    backends = [BackendType.TRANSFORMERS]  # Always available
    
    if check_vllm_available():
        backends.append(BackendType.VLLM)
    
    if check_llama_cpp_available():
        backends.append(BackendType.LLAMA_CPP)
    
    if check_tensorrt_llm_available():
        backends.append(BackendType.TENSORRT_LLM)
    
    return backends


__all__ = [
    "create_backend",
    "auto_select_backend",
    "get_available_backends",
    "check_vllm_available",
    "check_llama_cpp_available",
    "check_tensorrt_llm_available",
]













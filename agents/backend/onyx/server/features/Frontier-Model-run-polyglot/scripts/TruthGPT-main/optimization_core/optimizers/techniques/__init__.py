"""
Optimization Techniques Module

This module contains optimization techniques and computational optimizations.
"""

from __future__ import annotations

import importlib

__all__ = [
    'ComputationalOptimizer',
    'FusedAttention',
    'BatchOptimizer',
    'create_computational_optimizer',
    'TritonOptimizations',
    'TritonLayerNorm',
    'TritonLayerNormModule',
    'rotary_embed',
    'block_copy',
]

_LAZY_IMPORTS = {
    # Computational optimizations
    'ComputationalOptimizer': '..computational_optimizations',
    'FusedAttention': '..computational_optimizations',
    'BatchOptimizer': '..computational_optimizations',
    'create_computational_optimizer': '..computational_optimizations',
    # Triton optimizations
    'TritonOptimizations': '..triton_optimizations',
    'TritonLayerNorm': '..triton_optimizations',
    'TritonLayerNormModule': '..triton_optimizations',
    'rotary_embed': '..triton_optimizations',
    'block_copy': '..triton_optimizations',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for optimization techniques."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, package=__package__)
        obj = getattr(module, name)
        _import_cache[name] = obj
        return obj
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_techniques() -> list[str]:
    """List all available optimization techniques."""
    return list(_LAZY_IMPORTS.keys())


"""Performance Optimization"""

def generate_performance_code() -> str:
    return '''"""
Performance Optimization
=======================

Utilidades para optimización de performance.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def profile_model(model: nn.Module, input_shape: tuple, device: str = "cuda"):
    """Perfila modelo."""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("model_inference"):
            _ = model(dummy_input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total"))


def optimize_model(model: nn.Module) -> nn.Module:
    """Optimiza modelo con torch.jit."""
    model.eval()
    try:
        traced_model = torch.jit.trace(model, example_inputs=torch.randn(1, 512))
        logger.info("Model optimized with JIT")
        return traced_model
    except Exception as e:
        logger.warning(f"JIT optimization failed: {e}")
        return model


class MemoryOptimizer:
    """Optimizador de memoria."""
    
    @staticmethod
    def clear_cache():
        """Limpia cache de CUDA."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Obtiene uso de memoria."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9
            }
        return {}
'''


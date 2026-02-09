"""
Optimized Inference with torch.compile and other techniques
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Compile model with torch.compile for faster inference
    
    Args:
        model: Model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        
    Returns:
        Compiled model
    """
    if hasattr(torch, 'compile'):
        try:
            compiled = torch.compile(model, mode=mode)
            logger.info(f"Model compiled with mode={mode}")
            return compiled
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            return model
    else:
        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        return model


def optimize_for_inference_fast(model: nn.Module, example_input=None) -> nn.Module:
    """
    Fast optimization for inference
    
    Args:
        model: Model to optimize
        example_input: Example input for tracing
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # 1. Compile with torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = compile_model(model, mode="reduce-overhead")
    
    # 2. JIT trace if example input provided
    if example_input is not None:
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
                traced = torch.jit.optimize_for_inference(traced)
                logger.info("Model optimized with JIT")
                return traced
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")
    
    return model


class FastInferenceWrapper(nn.Module):
    """Wrapper for fast inference with caching"""
    
    def __init__(self, model: nn.Module, cache_size: int = 100):
        """
        Initialize fast inference wrapper
        
        Args:
            model: Base model
            cache_size: Cache size for results
        """
        super().__init__()
        self.model = model
        self.cache_size = cache_size
        self._cache = {}
        self._cache_keys = []
    
    def forward(self, *args, **kwargs):
        # Simple hash-based caching (for demonstration)
        # In production, use proper caching library
        cache_key = str(args) + str(kwargs)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self.model(*args, **kwargs)
        
        # Add to cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
        self._cache_keys.append(cache_key)
        
        return result


def create_fast_inference_model(
    model: nn.Module,
    example_input=None,
    use_compile: bool = True,
    use_cache: bool = False
) -> nn.Module:
    """
    Create optimized model for fast inference
    
    Args:
        model: Base model
        example_input: Example input for optimization
        use_compile: Use torch.compile
        use_cache: Use result caching
        
    Returns:
        Optimized model
    """
    if use_compile:
        model = optimize_for_inference_fast(model, example_input)
    
    if use_cache:
        model = FastInferenceWrapper(model)
    
    return model


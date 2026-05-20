"""
Profiler Utilities
===================

Code profiling for performance optimization.
"""

import torch
import time
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for PyTorch code."""
    
    def __init__(self, use_torch_profiler: bool = True):
        """
        Initialize profiler.
        
        Args:
            use_torch_profiler: Whether to use PyTorch profiler
        """
        self.use_torch_profiler = use_torch_profiler and PROFILER_AVAILABLE
        self.profiler = None
        self._logger = logger
    
    @contextmanager
    def profile(self, activities=None, record_shapes=False, profile_memory=False):
        """
        Context manager for profiling.
        
        Args:
            activities: Profiler activities
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory
        """
        if self.use_torch_profiler:
            activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            
            with profile(
                activities=activities,
                record_shapes=record_shapes,
                profile_memory=profile_memory
            ) as prof:
                yield prof
                prof.export_chrome_trace("trace.json")
                self._logger.info("Profiler trace saved to trace.json")
        else:
            # Simple timing
            start_time = time.time()
            yield None
            elapsed = time.time() - start_time
            self._logger.info(f"Execution time: {elapsed:.4f}s")
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile function.
        
        Args:
            func: Function to profile
        
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            self._logger.info(f"{func.__name__} took {elapsed:.4f}s")
            return result
        
        return wrapper


def profile_model_forward(model: torch.nn.Module, input_shape: tuple, device: str = "cpu"):
    """
    Profile model forward pass.
    
    Args:
        model: PyTorch model
        input_shape: Input shape
        device: Device
    
    Returns:
        Profiling results
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Profile
    profiler = PerformanceProfiler()
    
    with profiler.profile(record_shapes=True, profile_memory=True):
        with torch.no_grad():
            _ = model(dummy_input)
    
    return profiler





"""
Code Profiling

Utilities for profiling code performance and identifying bottlenecks.
"""

import logging
import time
import cProfile
import pstats
import io
from typing import Callable, Optional, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CodeProfiler:
    """Profile code execution time and identify bottlenecks."""
    
    def __init__(self):
        """Initialize code profiler."""
        self.profiler = cProfile.Profile()
    
    def profile(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Profile function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profile results
        """
        self.profiler.enable()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        self.profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        stats_str = s.getvalue()
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'stats': stats_str
        }
    
    def profile_model(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_warmup: int = 5,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model forward pass.
        
        Args:
            model: Model to profile
            input_tensor: Input tensor
            num_warmup: Number of warmup iterations
            num_iterations: Number of profiling iterations
            
        Returns:
            Profile results
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        self.profiler.enable()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        self.profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        stats_str = s.getvalue()
        
        avg_time = (end_time - start_time) / num_iterations
        
        return {
            'avg_forward_time': avg_time,
            'total_time': end_time - start_time,
            'stats': stats_str
        }


def profile_function(
    func: Callable,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to profile function.
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Profile results
    """
    profiler = CodeProfiler()
    return profiler.profile(func, *args, **kwargs)


def profile_model_forward(
    model: nn.Module,
    input_tensor: torch.Tensor,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to profile model forward pass.
    
    Args:
        model: Model to profile
        input_tensor: Input tensor
        **kwargs: Additional profiling arguments
        
    Returns:
        Profile results
    """
    profiler = CodeProfiler()
    return profiler.profile_model(model, input_tensor, **kwargs)




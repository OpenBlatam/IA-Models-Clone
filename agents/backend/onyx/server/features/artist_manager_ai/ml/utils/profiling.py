"""
Profiling Utilities
===================

Advanced profiling utilities for performance analysis.
"""

import torch
import time
import cProfile
import pstats
import io
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Performance profiler for PyTorch models.
    
    Features:
    - GPU profiling
    - CPU profiling
    - Memory profiling
    - Timing analysis
    """
    
    def __init__(self, use_cuda: bool = True):
        """
        Initialize profiler.
        
        Args:
            use_cuda: Whether to use CUDA profiling
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._logger = logger
    
    @contextmanager
    def profile(self, name: str = "operation"):
        """
        Context manager for profiling.
        
        Args:
            name: Operation name
        
        Yields:
            Profiler context
        """
        if self.use_cuda:
            with torch.cuda.profiler.profile() as prof:
                with torch.autograd.profiler.emit_nvtx():
                    start_time = time.time()
                    yield
                    end_time = time.time()
        else:
            start_time = time.time()
            yield
            end_time = time.time()
        
        duration = end_time - start_time
        self._logger.info(f"{name} took {duration:.4f} seconds")
    
    def profile_model(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model inference.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            num_runs: Number of runs for averaging
        
        Returns:
            Profiling results
        """
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        # Profile
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start_time = time.time()
                
                _ = model(input_tensor)
                
                if self.use_cuda:
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) / 1000.0)  # Convert to seconds
                else:
                    end_time = time.time()
                    times.append(end_time - start_time)
        
        # Memory profiling
        if self.use_cuda:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "throughput": 1.0 / np.mean(times)  # Samples per second
        }


class CodeProfiler:
    """
    Code profiler using cProfile.
    """
    
    def __init__(self):
        """Initialize code profiler."""
        self.profiler = cProfile.Profile()
        self._logger = logger
    
    @contextmanager
    def profile(self):
        """
        Context manager for code profiling.
        
        Yields:
            Profiler context
        """
        self.profiler.enable()
        yield
        self.profiler.disable()
    
    def get_stats(self, sort_by: str = "cumulative", num_stats: int = 20) -> str:
        """
        Get profiling statistics.
        
        Args:
            sort_by: Sort key
            num_stats: Number of stats to return
        
        Returns:
            Formatted stats string
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats(sort_by)
        ps.print_stats(num_stats)
        return s.getvalue()
    
    def save_stats(self, filename: str):
        """
        Save stats to file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            ps = pstats.Stats(self.profiler, stream=f)
            ps.sort_stats('cumulative')
            ps.print_stats()


class MemoryProfiler:
    """
    Memory profiler for tracking memory usage.
    """
    
    @staticmethod
    def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
        """
        Get memory statistics.
        
        Args:
            device: Device to check
        
        Returns:
            Memory stats in MB
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2
            }
        else:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": mem_info.rss / 1024**2,
                "vms_mb": mem_info.vms / 1024**2
            }
    
    @staticmethod
    def clear_cache(device: Optional[torch.device] = None):
        """
        Clear memory cache.
        
        Args:
            device: Device to clear
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


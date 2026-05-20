"""
Performance Profiling and Benchmarking
Profile code to identify bottlenecks
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import time
import cProfile
import pstats
import io
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ProfilingResult:
    """Result from profiling"""
    function_name: str
    total_time: float
    call_count: int
    average_time: float
    cumulative_time: float


class CodeProfiler:
    """
    Profile Python code to identify bottlenecks
    """
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.results: List[ProfilingResult] = []
    
    @contextmanager
    def profile(self, name: str = "operation"):
        """Context manager for profiling"""
        self.profiler.enable()
        start_time = time.time()
        try:
            yield
        finally:
            self.profiler.disable()
            end_time = time.time()
            
            # Get stats
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            logger.info(f"Profiling {name} (took {end_time - start_time:.4f}s)")
            logger.debug(s.getvalue())
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Profile a function call"""
        with self.profile(func.__name__):
            result = func(*args, **kwargs)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        return {
            "stats": s.getvalue(),
            "total_calls": ps.total_calls,
            "primitive_calls": ps.primitive_calls
        }


class PyTorchProfiler:
    """
    Profile PyTorch operations
    """
    
    def __init__(self, use_cuda: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PyTorch profiling")
        
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.profiler = None
    
    @contextmanager
    def profile(self, activities=None, record_shapes: bool = True):
        """Context manager for PyTorch profiling"""
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA if self.use_cuda else None
            ]
            activities = [a for a in activities if a is not None]
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=True
        ) as prof:
            yield prof
    
    def get_trace(self, profiler) -> str:
        """Get trace for Chrome tracing"""
        return profiler.export_chrome_trace("trace.json")
    
    def print_stats(self, profiler):
        """Print profiling statistics"""
        print(profiler.key_averages().table(
            sort_by="cuda_time_total" if self.use_cuda else "cpu_time_total",
            row_limit=20
        ))


class Benchmark:
    """
    Benchmark functions and models
    """
    
    @staticmethod
    def benchmark_function(
        func: Callable,
        *args,
        num_runs: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark a function"""
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            func(*args, **kwargs)
            times.append(time.time() - start)
        
        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "num_runs": num_runs
        }
    
    @staticmethod
    def benchmark_model(
        model: torch.nn.Module,
        input_shape: tuple,
        num_runs: int = 100,
        warmup: int = 10,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Benchmark a PyTorch model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for model benchmarking")
        
        model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Synchronize if CUDA
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(dummy_input)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                
                times.append(time.time() - start)
        
        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "num_runs": num_runs,
            "throughput": 1.0 / (sum(times) / len(times))  # samples per second
        }


class PerformanceMonitor:
    """
    Monitor system performance
    """
    
    @staticmethod
    def get_gpu_memory() -> Dict[str, Any]:
        """Get GPU memory usage"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"available": False}
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            "available": True,
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
            "total_gb": memory_total,
            "free_gb": memory_total - memory_reserved
        }
    
    @staticmethod
    def get_cpu_usage() -> Dict[str, Any]:
        """Get CPU usage"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "cpu_count": psutil.cpu_count()
            }
        except ImportError:
            return {"available": False}
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information"""
        info = {
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else None,
            "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
        
        info.update(PerformanceMonitor.get_gpu_memory())
        
        return info


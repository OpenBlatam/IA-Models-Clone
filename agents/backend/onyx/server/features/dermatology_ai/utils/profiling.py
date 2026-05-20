"""
Profiling and Performance Utilities
For identifying bottlenecks and optimizing code
"""

import torch
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


@contextmanager
def profile_region(name: str, enable: bool = True):
    """
    Context manager for profiling code regions
    
    Usage:
        with profile_region("forward_pass"):
            output = model(input)
    """
    if not enable:
        yield
        return
    
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        duration = end_time - start_time
        memory_delta = (end_memory - start_memory) / (1024 ** 2)  # MB
        
        logger.info(
            f"Profile [{name}]: "
            f"Time: {duration:.4f}s, "
            f"Memory: {memory_delta:+.2f}MB"
        )


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution
    
    Usage:
        @time_function
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.debug(f"{func.__name__} took {duration:.4f}s")
        return result
    return wrapper


class PerformanceMonitor:
    """
    Monitor performance metrics during training/inference
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, name: str):
        """Start timing a region"""
        self.start_times[name] = time.time()
    
    def end(self, name: str) -> float:
        """End timing a region and return duration"""
        if name not in self.start_times:
            logger.warning(f"Region '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        del self.start_times[name]
        
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a region"""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'total': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all regions"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()


def profile_model(
    model: torch.nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    device: str = "cuda",
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, Any]:
    """
    Profile model performance
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to use
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with profiling results
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Profile forward pass
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    # Memory usage
    if device == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'avg_forward_time_ms': avg_time * 1000,
        'throughput_fps': 1.0 / avg_time,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved,
        'total_params': param_count,
        'trainable_params': trainable_params,
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    }


def optimize_data_loader(
    data_loader: torch.utils.data.DataLoader,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> torch.utils.data.DataLoader:
    """
    Optimize DataLoader settings for better performance
    
    Args:
        data_loader: Original DataLoader
        num_workers: Number of worker processes (None = auto)
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        
    Returns:
        Optimized DataLoader
    """
    if num_workers is None:
        # Auto-detect optimal number of workers
        num_workers = min(4, torch.multiprocessing.cpu_count())
    
    return torch.utils.data.DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        shuffle=data_loader.sampler is None,
        sampler=data_loader.sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )


def check_gpu_utilization() -> Dict[str, Any]:
    """
    Check GPU utilization and memory usage
    
    Returns:
        Dictionary with GPU stats
    """
    if not torch.cuda.is_available():
        return {'available': False}
    
    stats = {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(),
        'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
        'memory_reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
        'max_memory_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'max_memory_reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2)
    }
    
    return stats


def clear_gpu_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")














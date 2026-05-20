"""
Performance Utilities
Utilities for monitoring and optimizing performance
"""

import time
import torch
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        if self.start_time is None:
            raise ValueError("Monitoring not started")
        
        elapsed = time.time() - self.start_time
        
        metrics = {
            'elapsed_time': elapsed,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
                'gpu_peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 ** 2)
            })
        
        self.metrics.update(metrics)
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()


@contextmanager
def performance_context(name: str = "operation", log: bool = True):
    """Context manager for performance monitoring"""
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        yield monitor
    finally:
        metrics = monitor.stop()
        if log:
            logger.info(f"{name} completed in {metrics['elapsed_time']:.4f}s")
            if torch.cuda.is_available():
                logger.info(
                    f"  GPU Memory: {metrics['gpu_peak_memory_mb']:.2f} MB "
                    f"(allocated: {metrics['gpu_memory_allocated_mb']:.2f} MB)"
                )


def benchmark_function(
    func: Callable,
    num_runs: int = 10,
    warmup_runs: int = 2,
    *args,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark a function
    
    Args:
        func: Function to benchmark
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean': sum(times) / len(times),
        'std': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        'min': min(times),
        'max': max(times),
        'total': sum(times)
    }


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
        'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
        'memory_percent': psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
        
        for i in range(torch.cuda.device_count()):
            info[f'gpu_{i}_memory_total_gb'] = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    else:
        info['cuda_available'] = False
    
    return info


def optimize_memory():
    """Optimize memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()


def clear_cache():
    """Clear all caches"""
    optimize_memory()
    
    # Clear Python cache
    import sys
    if hasattr(sys, 'getsizeof'):
        # Force garbage collection
        import gc
        gc.collect()














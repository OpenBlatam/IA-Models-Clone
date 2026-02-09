"""Benchmarking utilities."""

import time
from typing import Callable, Any, Dict
from functools import wraps
from contextlib import contextmanager


@contextmanager
def benchmark_context(name: str = "operation"):
    """
    Context manager for benchmarking.
    
    Args:
        name: Operation name
        
    Yields:
        Benchmark context
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.4f} seconds")


def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark function execution.
    
    Args:
        func: Function to benchmark
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.4f} seconds")
    
    return wrapper


def benchmark_async(func: Callable) -> Callable:
    """
    Decorator to benchmark async function execution.
    
    Args:
        func: Async function to benchmark
        
    Returns:
        Wrapped async function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.4f} seconds")
    
    return wrapper


class Benchmark:
    """Benchmark manager."""
    
    def __init__(self):
        self.results: Dict[str, list] = {}
    
    def record(self, name: str, duration: float) -> None:
        """
        Record benchmark result.
        
        Args:
            name: Benchmark name
            duration: Duration in seconds
        """
        if name not in self.results:
            self.results[name] = []
        self.results[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for benchmark.
        
        Args:
            name: Benchmark name
            
        Returns:
            Statistics dictionary
        """
        if name not in self.results or not self.results[name]:
            return {}
        
        durations = self.results[name]
        
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
            "total": sum(durations),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all benchmarks.
        
        Returns:
            Dictionary of benchmark statistics
        """
        return {name: self.get_stats(name) for name in self.results.keys()}


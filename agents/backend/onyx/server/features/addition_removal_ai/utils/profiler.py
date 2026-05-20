"""
Performance Profiling Utilities
"""

import torch
import time
from typing import Dict, Optional, Callable
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for training and inference"""
    
    def __init__(self):
        """Initialize profiler"""
        self.metrics = {}
        self.timers = {}
    
    @contextmanager
    def profile(self, name: str):
        """Profile a code block"""
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(elapsed)
    
    def get_stats(self, name: str) -> Dict:
        """Get statistics for a profiled operation"""
        if name not in self.metrics:
            return {}
        
        times = self.metrics[name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all profiled operations"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.timers = {}
    
    def print_summary(self):
        """Print profiling summary"""
        print("\n=== Performance Profile ===")
        for name, stats in self.get_all_stats().items():
            print(f"\n{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['total']:.4f}s")
            print(f"  Mean: {stats['mean']:.4f}s")
            print(f"  Min: {stats['min']:.4f}s")
            print(f"  Max: {stats['max']:.4f}s")
            print(f"  Std: {stats['std']:.4f}s")


def profile_model(
    model: torch.nn.Module,
    input_shape: tuple,
    num_runs: int = 100,
    warmup: int = 10,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Profile model inference
    
    Args:
        model: Model to profile
        input_shape: Input shape
        num_runs: Number of runs
        warmup: Warmup runs
        device: Device to use
        
    Returns:
        Profiling results
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Profile
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    return {
        "mean": sum(times) / len(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        "min": min(times),
        "max": max(times),
        "fps": 1.0 / (sum(times) / len(times))
    }


def create_profiler() -> PerformanceProfiler:
    """Factory function to create profiler"""
    return PerformanceProfiler()


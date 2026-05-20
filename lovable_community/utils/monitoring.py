"""
Advanced Monitoring and Profiling

Comprehensive monitoring for training and inference.
"""

import logging
import time
import torch
from typing import Dict, Any, Optional, List
from collections import defaultdict
from contextlib import contextmanager
import psutil
import os

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Advanced performance monitor for training and inference.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        self.counters = defaultdict(int)
        self.memory_stats = []
    
    def record(self, name: str, value: float) -> None:
        """Record a metric value."""
        self.metrics[name].append(value)
    
    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and return duration."""
        if name not in self.timers:
            logger.warning(f"Timer {name} not started")
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.record(f"{name}_time", duration)
        del self.timers[name]
        return duration
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] += value
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "total": sum(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def record_memory(self) -> Dict[str, float]:
        """Record current memory usage."""
        stats = {}
        
        # CPU memory
        process = psutil.Process(os.getpid())
        stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[f"gpu_{i}_allocated_mb"] = (
                    torch.cuda.memory_allocated(i) / 1024 / 1024
                )
                stats[f"gpu_{i}_reserved_mb"] = (
                    torch.cuda.memory_reserved(i) / 1024 / 1024
                )
        
        self.memory_stats.append(stats)
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()
        self.memory_stats.clear()


class TrainingProfiler:
    """
    Profiler for training loops.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.monitor = PerformanceMonitor()
        self.current_epoch = 0
        self.current_step = 0
    
    def start_epoch(self, epoch: int) -> None:
        """Start profiling an epoch."""
        if not self.enabled:
            return
        
        self.current_epoch = epoch
        self.monitor.start_timer("epoch")
        self.monitor.record_memory()
    
    def end_epoch(self) -> Dict[str, Any]:
        """End profiling an epoch."""
        if not self.enabled:
            return {}
        
        epoch_time = self.monitor.end_timer("epoch")
        memory = self.monitor.record_memory()
        
        return {
            "epoch": self.current_epoch,
            "epoch_time": epoch_time,
            "memory": memory
        }
    
    def start_step(self, step: int) -> None:
        """Start profiling a step."""
        if not self.enabled:
            return
        
        self.current_step = step
        self.monitor.start_timer("step")
    
    def end_step(self, loss: float) -> Dict[str, Any]:
        """End profiling a step."""
        if not self.enabled:
            return {}
        
        step_time = self.monitor.end_timer("step")
        self.monitor.record("loss", loss)
        self.monitor.record("step_time", step_time)
        
        return {
            "step": self.current_step,
            "step_time": step_time,
            "loss": loss
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.enabled:
            return {}
        
        return {
            "metrics": self.monitor.get_all_stats(),
            "counters": dict(self.monitor.counters),
            "memory_stats": self.monitor.memory_stats[-10:]  # Last 10
        }


@contextmanager
def profile_operation(name: str, monitor: Optional[PerformanceMonitor] = None):
    """
    Context manager for profiling an operation.
    
    Args:
        name: Operation name
        monitor: Optional monitor instance
    """
    if monitor is None:
        monitor = PerformanceMonitor()
    
    with monitor.timer(name):
        yield


def log_gpu_memory(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Log current GPU memory usage.
    
    Args:
        device: Device to check (None for all)
        
    Returns:
        Dictionary with memory stats
    """
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    
    if device is not None and device.type == "cuda":
        idx = device.index
        stats[f"gpu_{idx}_allocated_mb"] = (
            torch.cuda.memory_allocated(idx) / 1024 / 1024
        )
        stats[f"gpu_{idx}_reserved_mb"] = (
            torch.cuda.memory_reserved(idx) / 1024 / 1024
        )
    else:
        for i in range(torch.cuda.device_count()):
            stats[f"gpu_{i}_allocated_mb"] = (
                torch.cuda.memory_allocated(i) / 1024 / 1024
            )
            stats[f"gpu_{i}_reserved_mb"] = (
                torch.cuda.memory_reserved(i) / 1024 / 1024
            )
    
    return stats














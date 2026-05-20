"""
Performance Profiler Module
==========================

Profile training performance and identify bottlenecks.

Author: BUL System
Date: 2024
"""

import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profile training performance and identify bottlenecks.
    
    Tracks:
    - Time spent in different operations
    - Memory usage patterns
    - Throughput metrics
    - Bottleneck identification
    
    Example:
        >>> profiler = PerformanceProfiler()
        >>> profiler.start()
        >>> with profiler.profile("data_loading"):
        ...     load_data()
        >>> with profiler.profile("model_forward"):
        ...     model(inputs)
        >>> report = profiler.get_report()
    """
    
    def __init__(self):
        """Initialize PerformanceProfiler."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_time: Optional[float] = None
        self.total_steps: int = 0
        self.current_step: int = 0
    
    def start(self) -> None:
        """Start profiling."""
        self.start_time = time.time()
        self.total_steps = 0
        self.current_step = 0
        logger.info("Performance profiling started")
    
    @contextmanager
    def profile(self, operation_name: str):
        """
        Context manager to profile an operation.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.timings[operation_name].append(duration)
            logger.debug(f"{operation_name}: {duration:.4f}s")
    
    def record_step(self, step: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Record a training step.
        
        Args:
            step: Step number
            metrics: Optional metrics for this step
        """
        self.current_step = step
        self.total_steps = step
    
    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing summary for all operations.
        
        Returns:
            Dictionary with timing statistics per operation
        """
        summary = {}
        
        for operation, timings in self.timings.items():
            if not timings:
                continue
            
            summary[operation] = {
                "total_time": sum(timings),
                "avg_time": sum(timings) / len(timings),
                "min_time": min(timings),
                "max_time": max(timings),
                "count": len(timings),
                "total_percentage": (sum(timings) / (time.time() - self.start_time) * 100) if self.start_time else 0,
            }
        
        return summary
    
    def get_bottlenecks(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold: Minimum percentage of total time to be considered a bottleneck
            
        Returns:
            List of bottlenecks sorted by impact
        """
        if not self.start_time:
            return []
        
        total_time = time.time() - self.start_time
        bottlenecks = []
        
        for operation, timings in self.timings.items():
            if not timings:
                continue
            
            operation_time = sum(timings)
            percentage = (operation_time / total_time * 100) if total_time > 0 else 0
            
            if percentage >= threshold * 100:
                bottlenecks.append({
                    "operation": operation,
                    "total_time": operation_time,
                    "percentage": percentage,
                    "avg_time": operation_time / len(timings),
                    "count": len(timings),
                })
        
        # Sort by percentage (highest impact first)
        bottlenecks.sort(key=lambda x: x["percentage"], reverse=True)
        return bottlenecks
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get complete profiling report.
        
        Returns:
            Dictionary with complete profiling information
        """
        if not self.start_time:
            return {"error": "Profiling not started"}
        
        total_time = time.time() - self.start_time
        
        return {
            "total_time": total_time,
            "total_steps": self.total_steps,
            "steps_per_second": self.total_steps / total_time if total_time > 0 else 0,
            "timing_summary": self.get_timing_summary(),
            "bottlenecks": self.get_bottlenecks(),
        }
    
    def reset(self) -> None:
        """Reset profiling data."""
        self.timings.clear()
        self.start_time = None
        self.total_steps = 0
        self.current_step = 0


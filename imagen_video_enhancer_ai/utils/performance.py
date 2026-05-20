"""
Performance utilities for Imagen Video Enhancer AI
==================================================

Performance monitoring and optimization utilities.
"""

import time
import logging
from typing import Any, Dict
from contextlib import asynccontextmanager

# Import common decorators
from ..core.decorators import measure_time

logger = logging.getLogger(__name__)


@asynccontextmanager
async def measure_context(operation_name: str):
    """
    Context manager to measure execution time.
    
    Usage:
        async with measure_context("my_operation"):
            # code to measure
            ...
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.debug(f"{operation_name} took {duration:.3f}s")


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(self, operation: str, duration: float):
        """Record a performance metric."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        return {
            "count": len(durations),
            "total": sum(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            operation: self.get_stats(operation)
            for operation in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


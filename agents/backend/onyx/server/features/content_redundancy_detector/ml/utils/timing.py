"""
Timing Utilities
Time measurement and profiling utilities
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Timer:
    """
    Timer for measuring execution time
    """
    
    def __init__(self):
        """Initialize timer"""
        self.times = defaultdict(list)
        self.starts = {}
    
    def start(self, name: str) -> None:
        """
        Start timer for operation
        
        Args:
            name: Operation name
        """
        self.starts[name] = time.time()
    
    def stop(self, name: str) -> float:
        """
        Stop timer for operation
        
        Args:
            name: Operation name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.starts:
            logger.warning(f"Timer {name} was not started")
            return 0.0
        
        elapsed = time.time() - self.starts[name]
        self.times[name].append(elapsed)
        del self.starts[name]
        return elapsed
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for operation
        
        Args:
            name: Operation name
            
        Returns:
            Dictionary with statistics
        """
        if name not in self.times or not self.times[name]:
            return {}
        
        times = self.times[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operations
        
        Returns:
            Dictionary with all statistics
        """
        return {name: self.get_stats(name) for name in self.times}
    
    @contextmanager
    def time_block(self, name: str):
        """
        Context manager for timing code blocks
        
        Args:
            name: Operation name
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)


class PerformanceTimer:
    """
    Performance timer with detailed metrics
    """
    
    def __init__(self):
        """Initialize performance timer"""
        self.timer = Timer()
        self.memory_usage = defaultdict(list)
    
    def record_memory(self, name: str) -> None:
        """
        Record memory usage
        
        Args:
            name: Operation name
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 ** 2)
            self.memory_usage[name].append(memory_mb)
        except ImportError:
            logger.warning("psutil not available for memory tracking")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'timing': self.timer.get_all_stats(),
            'memory': dict(self.memory_usage),
        }




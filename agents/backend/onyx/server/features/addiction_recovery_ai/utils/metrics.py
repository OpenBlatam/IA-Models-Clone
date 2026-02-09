"""
Metrics utilities
Performance and metrics tracking functions
"""

from typing import Dict, List, Optional, Callable
from collections import defaultdict
from datetime import datetime
import time


class Counter:
    """
    Simple counter metric
    """
    
    def __init__(self, name: str):
        """
        Initialize counter
        
        Args:
            name: Counter name
        """
        self.name = name
        self.value = 0
    
    def increment(self, amount: int = 1) -> None:
        """
        Increment counter
        
        Args:
            amount: Amount to increment
        """
        self.value += amount
    
    def decrement(self, amount: int = 1) -> None:
        """
        Decrement counter
        
        Args:
            amount: Amount to decrement
        """
        self.value -= amount
    
    def reset(self) -> None:
        """
        Reset counter
        """
        self.value = 0
    
    def get(self) -> int:
        """
        Get counter value
        
        Returns:
            Counter value
        """
        return self.value


class Timer:
    """
    Timer metric
    """
    
    def __init__(self, name: str):
        """
        Initialize timer
        
        Args:
            name: Timer name
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.total_time = 0.0
        self.count = 0
    
    def start(self) -> None:
        """
        Start timer
        """
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop timer
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        self.total_time += elapsed
        self.count += 1
        self.start_time = None
        
        return elapsed
    
    def get_average(self) -> float:
        """
        Get average time
        
        Returns:
            Average time in seconds
        """
        if self.count == 0:
            return 0.0
        
        return self.total_time / self.count
    
    def reset(self) -> None:
        """
        Reset timer
        """
        self.start_time = None
        self.total_time = 0.0
        self.count = 0


class MetricsCollector:
    """
    Metrics collector
    """
    
    def __init__(self):
        """
        Initialize metrics collector
        """
        self.counters: Dict[str, Counter] = {}
        self.timers: Dict[str, Timer] = {}
        self.gauges: Dict[str, float] = {}
    
    def get_counter(self, name: str) -> Counter:
        """
        Get or create counter
        
        Args:
            name: Counter name
        
        Returns:
            Counter instance
        """
        if name not in self.counters:
            self.counters[name] = Counter(name)
        return self.counters[name]
    
    def get_timer(self, name: str) -> Timer:
        """
        Get or create timer
        
        Args:
            name: Timer name
        
        Returns:
            Timer instance
        """
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]
    
    def set_gauge(self, name: str, value: float) -> None:
        """
        Set gauge value
        
        Args:
            name: Gauge name
            value: Gauge value
        """
        self.gauges[name] = value
    
    def get_gauge(self, name: str, default: float = 0.0) -> float:
        """
        Get gauge value
        
        Args:
            name: Gauge name
            default: Default value
        
        Returns:
            Gauge value
        """
        return self.gauges.get(name, default)
    
    def get_all_metrics(self) -> Dict:
        """
        Get all metrics
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "counters": {name: counter.get() for name, counter in self.counters.items()},
            "timers": {
                name: {
                    "total": timer.total_time,
                    "count": timer.count,
                    "average": timer.get_average()
                }
                for name, timer in self.timers.items()
            },
            "gauges": self.gauges.copy()
        }
    
    def reset(self) -> None:
        """
        Reset all metrics
        """
        for counter in self.counters.values():
            counter.reset()
        for timer in self.timers.values():
            timer.reset()
        self.gauges.clear()


def track_time(func: Callable) -> Callable:
    """
    Decorator to track function execution time
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{func.__name__} executed in {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"{func.__name__} failed after {elapsed:.4f}s: {str(e)}")
            raise
    return wrapper


def calculate_rate(count: int, duration: float) -> float:
    """
    Calculate rate (count per second)
    
    Args:
        count: Count value
        duration: Duration in seconds
    
    Returns:
        Rate per second
    """
    if duration == 0:
        return 0.0
    
    return count / duration


def calculate_percentage(part: float, total: float) -> float:
    """
    Calculate percentage
    
    Args:
        part: Part value
        total: Total value
    
    Returns:
        Percentage
    """
    if total == 0:
        return 0.0
    
    return (part / total) * 100


def calculate_average(values: List[float]) -> float:
    """
    Calculate average
    
    Args:
        values: List of values
    
    Returns:
        Average value
    """
    if not values:
        return 0.0
    
    return sum(values) / len(values)


def calculate_throughput(operations: int, time_seconds: float) -> float:
    """
    Calculate throughput (operations per second)
    
    Args:
        operations: Number of operations
        time_seconds: Time in seconds
    
    Returns:
        Throughput (ops/sec)
    """
    if time_seconds == 0:
        return 0.0
    
    return operations / time_seconds

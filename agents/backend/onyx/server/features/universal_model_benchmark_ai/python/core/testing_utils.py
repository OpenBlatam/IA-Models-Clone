"""
Testing Utilities Module - Testing helpers and fixtures.

Provides:
- Test fixtures
- Mock data generators
- Assertion helpers
- Performance testing utilities
"""

import logging
import random
import string
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MockDataGenerator:
    """Mock data generator."""
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate random string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_int(min_val: int = 0, max_val: int = 100) -> int:
        """Generate random integer."""
        return random.randint(min_val, max_val)
    
    @staticmethod
    def random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate random float."""
        return random.uniform(min_val, max_val)
    
    @staticmethod
    def random_benchmark_result(model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate random benchmark result."""
        return {
            "benchmark_name": f"benchmark_{MockDataGenerator.random_string(5)}",
            "model_name": model_name or f"model_{MockDataGenerator.random_string(5)}",
            "accuracy": MockDataGenerator.random_float(0.5, 1.0),
            "latency_p50": MockDataGenerator.random_float(0.1, 1.0),
            "latency_p95": MockDataGenerator.random_float(0.2, 2.0),
            "latency_p99": MockDataGenerator.random_float(0.3, 3.0),
            "throughput": MockDataGenerator.random_float(50.0, 200.0),
            "memory_usage": {
                "gpu": MockDataGenerator.random_float(4.0, 16.0),
                "cpu": MockDataGenerator.random_float(2.0, 8.0),
            },
            "total_samples": MockDataGenerator.random_int(100, 1000),
            "correct_samples": MockDataGenerator.random_int(50, 950),
            "timestamp": datetime.now().isoformat(),
        }
    
    @staticmethod
    def random_experiment_config() -> Dict[str, Any]:
        """Generate random experiment config."""
        return {
            "name": f"experiment_{MockDataGenerator.random_string(8)}",
            "description": f"Test experiment {MockDataGenerator.random_string(20)}",
            "model_name": f"model_{MockDataGenerator.random_string(5)}",
            "benchmark_name": f"benchmark_{MockDataGenerator.random_string(5)}",
            "hyperparameters": {
                "temperature": MockDataGenerator.random_float(0.1, 1.0),
                "top_p": MockDataGenerator.random_float(0.5, 1.0),
                "max_tokens": MockDataGenerator.random_int(100, 1000),
            },
            "tags": [MockDataGenerator.random_string(5) for _ in range(3)],
        }


class PerformanceTestHelper:
    """Performance testing helper."""
    
    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """
        Measure function execution time.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time)
        """
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    
    @staticmethod
    def benchmark_function(
        func: Callable,
        iterations: int = 100,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Benchmark function performance.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Performance statistics
        """
        import time
        import statistics
        
        times = []
        for _ in range(iterations):
            start = time.time()
            func(*args, **kwargs)
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            "iterations": iterations,
            "total_time": sum(times),
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
        }


class AssertionHelper:
    """Assertion helper utilities."""
    
    @staticmethod
    def assert_approximately_equal(
        actual: float,
        expected: float,
        tolerance: float = 0.01,
        message: Optional[str] = None,
    ) -> None:
        """
        Assert two floats are approximately equal.
        
        Args:
            actual: Actual value
            expected: Expected value
            tolerance: Tolerance
            message: Optional error message
        """
        diff = abs(actual - expected)
        if diff > tolerance:
            msg = message or f"Values not approximately equal: {actual} != {expected} (tolerance: {tolerance})"
            raise AssertionError(msg)
    
    @staticmethod
    def assert_in_range(
        value: float,
        min_val: float,
        max_val: float,
        message: Optional[str] = None,
    ) -> None:
        """
        Assert value is in range.
        
        Args:
            value: Value to check
            min_val: Minimum value
            max_val: Maximum value
            message: Optional error message
        """
        if not (min_val <= value <= max_val):
            msg = message or f"Value {value} not in range [{min_val}, {max_val}]"
            raise AssertionError(msg)
    
    @staticmethod
    def assert_dict_contains(
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        message: Optional[str] = None,
    ) -> None:
        """
        Assert dictionary contains expected keys and values.
        
        Args:
            actual: Actual dictionary
            expected: Expected dictionary
            message: Optional error message
        """
        for key, value in expected.items():
            if key not in actual:
                msg = message or f"Key '{key}' not found in dictionary"
                raise AssertionError(msg)
            if actual[key] != value:
                msg = message or f"Value for key '{key}' mismatch: {actual[key]} != {value}"
                raise AssertionError(msg)













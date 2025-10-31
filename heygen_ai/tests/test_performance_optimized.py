"""
Optimized performance tests for HeyGen AI system.
Provides comprehensive performance testing with advanced optimizations.
"""

import pytest
import time
import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable, Optional
import statistics
import psutil
import gc
from dataclasses import dataclass
from pathlib import Path
import json
import random

def cpu_intensive_task(n):
    """CPU intensive task for multiprocessing tests."""
    return sum(i * i for i in range(n))

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    iterations: int
    throughput: float
    timestamp: float

class PerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    def measure_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an operation."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.perf_counter()
        start_cpu = self.process.cpu_percent()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_cpu = self.process.cpu_percent()
        
        # Calculate metrics
        duration = end_time - start_time
        memory_usage = (self.process.memory_info().rss / 1024 / 1024) - initial_memory
        cpu_usage = (start_cpu + end_cpu) / 2
        iterations = 1
        throughput = 1 / duration if duration > 0 else float('inf')
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            iterations=iterations,
            throughput=throughput,
            timestamp=time.time()
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def measure_iterations(self, operation_name: str, func: Callable, iterations: int, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance over multiple iterations."""
        # Force garbage collection
        gc.collect()
        
        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.perf_counter()
        start_cpu = self.process.cpu_percent()
        
        # Execute the function multiple times
        for _ in range(iterations):
            func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_cpu = self.process.cpu_percent()
        
        # Calculate metrics
        total_duration = end_time - start_time
        avg_duration = total_duration / iterations
        memory_usage = (self.process.memory_info().rss / 1024 / 1024) - initial_memory
        cpu_usage = (start_cpu + end_cpu) / 2
        throughput = iterations / total_duration if total_duration > 0 else float('inf')
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=avg_duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            iterations=iterations,
            throughput=throughput,
            timestamp=time.time()
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        durations = [m.duration for m in self.metrics]
        memory_usage = [m.memory_usage for m in self.metrics]
        cpu_usage = [m.cpu_usage for m in self.metrics]
        throughput = [m.throughput for m in self.metrics]
        
        return {
            "total_operations": len(self.metrics),
            "total_duration": sum(durations),
            "average_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
            "total_memory_usage": sum(memory_usage),
            "average_memory_usage": statistics.mean(memory_usage),
            "max_memory_usage": max(memory_usage),
            "average_cpu_usage": statistics.mean(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "average_throughput": statistics.mean(throughput),
            "max_throughput": max(throughput)
        }

# Performance test fixtures
@pytest.fixture
def profiler():
    """Performance profiler fixture."""
    return PerformanceProfiler()

@pytest.fixture
def performance_data():
    """Performance test data fixture."""
    return {
        "small_list": list(range(100)),
        "medium_list": list(range(1000)),
        "large_list": list(range(10000)),
        "small_dict": {f"key_{i}": f"value_{i}" for i in range(100)},
        "medium_dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
        "large_dict": {f"key_{i}": f"value_{i}" for i in range(10000)},
        "json_data": {
            "users": [
                {
                    "id": i,
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "active": i % 2 == 0,
                    "metadata": {
                        "created_at": "2024-01-01T00:00:00Z",
                        "last_login": "2024-12-21T12:00:00Z",
                        "permissions": ["read", "write"] if i % 3 == 0 else ["read"]
                    }
                }
                for i in range(1000)
            ]
        }
    }

# Performance test classes
class TestBasicPerformance:
    """Basic performance tests."""
    
    def test_list_operations_performance(self, profiler, performance_data):
        """Test list operations performance."""
        # Test list sorting
        metrics = profiler.measure_iterations(
            "list_sorting",
            lambda: sorted(performance_data["large_list"]),
            100
        )
        
        assert metrics.duration < 0.1, f"List sorting too slow: {metrics.duration:.3f}s"
        assert metrics.throughput > 100, f"Throughput too low: {metrics.throughput:.1f} ops/s"
    
    def test_dict_operations_performance(self, profiler, performance_data):
        """Test dictionary operations performance."""
        # Test dictionary access
        def dict_access_test():
            data = performance_data["large_dict"]
            return [data[f"key_{i}"] for i in range(100)]
        
        metrics = profiler.measure_iterations(
            "dict_access",
            dict_access_test,
            50
        )
        
        assert metrics.duration < 0.05, f"Dict access too slow: {metrics.duration:.3f}s"
        assert metrics.throughput > 200, f"Throughput too low: {metrics.throughput:.1f} ops/s"
    
    def test_string_operations_performance(self, profiler):
        """Test string operations performance."""
        test_string = "This is a test string for performance testing"
        
        def string_operations():
            return test_string.upper().lower().replace("test", "performance").split()
        
        metrics = profiler.measure_iterations(
            "string_operations",
            string_operations,
            1000
        )
        
        assert metrics.duration < 0.01, f"String operations too slow: {metrics.duration:.3f}s"
        assert metrics.throughput > 1000, f"Throughput too low: {metrics.throughput:.1f} ops/s"

class TestConcurrencyPerformance:
    """Concurrency performance tests."""
    
    def test_threading_performance(self, profiler):
        """Test threading performance."""
        def cpu_intensive_task(n):
            return sum(i * i for i in range(n))
        
        def threading_test():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cpu_intensive_task, 1000) for _ in range(10)]
                return [future.result() for future in futures]
        
        metrics = profiler.measure_operation("threading_test", threading_test)
        
        assert metrics.duration < 1.0, f"Threading test too slow: {metrics.duration:.3f}s"
        assert metrics.memory_usage < 50, f"Memory usage too high: {metrics.memory_usage:.1f} MB"
    
    def test_async_performance(self, profiler):
        """Test async performance."""
        async def async_task(delay):
            await asyncio.sleep(delay)
            return delay * 2
        
        async def async_test():
            tasks = [async_task(0.001) for _ in range(100)]
            return await asyncio.gather(*tasks)
        
        def run_async_test():
            return asyncio.run(async_test())
        
        metrics = profiler.measure_operation("async_test", run_async_test)
        
        assert metrics.duration < 0.5, f"Async test too slow: {metrics.duration:.3f}s"
        assert metrics.throughput > 30, f"Throughput too low: {metrics.throughput:.1f} ops/s"
    
    def test_multiprocessing_performance(self, profiler):
        """Test multiprocessing performance."""
        def multiprocessing_test():
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(cpu_intensive_task, 1000) for _ in range(4)]
                return [future.result() for future in futures]
        
        metrics = profiler.measure_operation("multiprocessing_test", multiprocessing_test)
        
        assert metrics.duration < 2.0, f"Multiprocessing test too slow: {metrics.duration:.3f}s"
        assert metrics.memory_usage < 100, f"Memory usage too high: {metrics.memory_usage:.1f} MB"

class TestMemoryPerformance:
    """Memory performance tests."""
    
    def test_memory_allocation_performance(self, profiler):
        """Test memory allocation performance."""
        def memory_allocation_test():
            # Allocate large amounts of memory
            data = []
            for i in range(1000):
                data.append([j for j in range(100)])
            return len(data)
        
        metrics = profiler.measure_operation("memory_allocation", memory_allocation_test)
        
        assert metrics.memory_usage < 200, f"Memory usage too high: {metrics.memory_usage:.1f} MB"
        assert metrics.duration < 1.0, f"Memory allocation too slow: {metrics.duration:.3f}s"
    
    def test_memory_cleanup_performance(self, profiler):
        """Test memory cleanup performance."""
        def memory_cleanup_test():
            # Allocate and deallocate memory
            data = [i for i in range(10000)]
            del data
            gc.collect()
            return True
        
        metrics = profiler.measure_iterations(
            "memory_cleanup",
            memory_cleanup_test,
            100
        )
        
        assert metrics.memory_usage < 10, f"Memory usage too high: {metrics.memory_usage:.1f} MB"
        assert metrics.duration < 0.5, f"Memory cleanup too slow: {metrics.duration:.3f}s"

class TestIOPerformance:
    """I/O performance tests."""
    
    def test_json_serialization_performance(self, profiler, performance_data):
        """Test JSON serialization performance."""
        def json_serialization_test():
            return json.dumps(performance_data["json_data"])
        
        metrics = profiler.measure_iterations(
            "json_serialization",
            json_serialization_test,
            100
        )
        
        assert metrics.duration < 0.1, f"JSON serialization too slow: {metrics.duration:.3f}s"
        assert metrics.throughput > 100, f"Throughput too low: {metrics.throughput:.1f} ops/s"
    
    def test_json_deserialization_performance(self, profiler, performance_data):
        """Test JSON deserialization performance."""
        json_string = json.dumps(performance_data["json_data"])
        
        def json_deserialization_test():
            return json.loads(json_string)
        
        metrics = profiler.measure_iterations(
            "json_deserialization",
            json_deserialization_test,
            100
        )
        
        assert metrics.duration < 0.1, f"JSON deserialization too slow: {metrics.duration:.3f}s"
        assert metrics.throughput > 50, f"Throughput too low: {metrics.throughput:.1f} ops/s"
    
    def test_file_operations_performance(self, profiler, tmp_path):
        """Test file operations performance."""
        test_file = tmp_path / "test_performance.txt"
        test_data = "This is test data for performance testing.\n" * 1000
        
        def file_write_test():
            test_file.write_text(test_data)
        
        def file_read_test():
            return test_file.read_text()
        
        # Test write performance
        write_metrics = profiler.measure_operation("file_write", file_write_test)
        assert write_metrics.duration < 0.1, f"File write too slow: {write_metrics.duration:.3f}s"
        
        # Test read performance
        read_metrics = profiler.measure_operation("file_read", file_read_test)
        assert read_metrics.duration < 0.1, f"File read too slow: {read_metrics.duration:.3f}s"

class TestAlgorithmPerformance:
    """Algorithm performance tests."""
    
    def test_sorting_algorithm_performance(self, profiler):
        """Test different sorting algorithms performance."""
        test_data = [random.randint(1, 1000) for _ in range(1000)]
        
        # Test built-in sort
        def builtin_sort():
            return sorted(test_data)
        
        builtin_metrics = profiler.measure_operation("builtin_sort", builtin_sort)
        
        # Test bubble sort (inefficient for comparison)
        def bubble_sort(arr):
            n = len(arr)
            arr = arr.copy()
            for i in range(n):
                for j in range(0, n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr
        
        bubble_metrics = profiler.measure_operation("bubble_sort", lambda: bubble_sort(test_data))
        
        # Built-in sort should be much faster
        assert builtin_metrics.duration < bubble_metrics.duration, "Built-in sort should be faster than bubble sort"
        assert builtin_metrics.duration < 0.01, f"Built-in sort too slow: {builtin_metrics.duration:.3f}s"
    
    def test_search_algorithm_performance(self, profiler):
        """Test search algorithms performance."""
        sorted_data = sorted([random.randint(1, 1000) for _ in range(1000)])
        target = sorted_data[500]  # Middle element
        
        # Test linear search
        def linear_search(arr, target):
            for i, val in enumerate(arr):
                if val == target:
                    return i
            return -1
        
        linear_metrics = profiler.measure_iterations(
            "linear_search",
            lambda: linear_search(sorted_data, target),
            100
        )
        
        # Test binary search
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        
        binary_metrics = profiler.measure_iterations(
            "binary_search",
            lambda: binary_search(sorted_data, target),
            100
        )
        
        # Binary search should be faster
        assert binary_metrics.duration < linear_metrics.duration, "Binary search should be faster than linear search"
        assert binary_metrics.duration < 0.001, f"Binary search too slow: {binary_metrics.duration:.3f}s"

class TestSystemPerformance:
    """System-level performance tests."""
    
    def test_system_resource_usage(self, profiler):
        """Test system resource usage."""
        def resource_intensive_task():
            # CPU intensive task
            result = 0
            for i in range(100000):
                result += i * i
            return result
        
        metrics = profiler.measure_operation("resource_intensive_task", resource_intensive_task)
        
        # Check resource usage is reasonable
        assert metrics.cpu_usage < 100, f"CPU usage too high: {metrics.cpu_usage:.1f}%"
        assert metrics.memory_usage < 50, f"Memory usage too high: {metrics.memory_usage:.1f} MB"
        assert metrics.duration < 1.0, f"Task too slow: {metrics.duration:.3f}s"
    
    def test_garbage_collection_performance(self, profiler):
        """Test garbage collection performance."""
        def gc_test():
            # Create objects that will be garbage collected
            data = []
            for i in range(1000):
                data.append([j for j in range(100)])
            del data
            gc.collect()
            return True
        
        metrics = profiler.measure_iterations(
            "garbage_collection",
            gc_test,
            10
        )
        
        assert metrics.duration < 1.0, f"GC too slow: {metrics.duration:.3f}s"
        assert metrics.memory_usage < 20, f"Memory usage too high: {metrics.memory_usage:.1f} MB"

# Performance test utilities
def test_performance_summary(profiler):
    """Test performance summary generation."""
    # Run some operations to generate metrics
    profiler.measure_operation("test_op1", lambda: sum(range(1000)))
    profiler.measure_operation("test_op2", lambda: sorted([random.randint(1, 100) for _ in range(100)]))
    
    summary = profiler.get_summary()
    
    assert "total_operations" in summary
    assert "average_duration" in summary
    assert "total_memory_usage" in summary
    assert summary["total_operations"] == 2
    assert summary["average_duration"] > 0
    assert summary["total_memory_usage"] >= 0

def test_performance_benchmark():
    """Test performance benchmarking."""
    profiler = PerformanceProfiler()
    
    # Benchmark different operations
    operations = [
        ("sum_range", lambda: sum(range(1000))),
        ("list_comprehension", lambda: [i * i for i in range(1000)]),
        ("dict_comprehension", lambda: {i: i * i for i in range(1000)}),
        ("string_join", lambda: "".join([str(i) for i in range(1000)])),
    ]
    
    for name, operation in operations:
        profiler.measure_iterations(name, operation, 100)
    
    summary = profiler.get_summary()
    
    # All operations should complete successfully
    assert summary["total_operations"] == len(operations)
    assert summary["average_duration"] > 0
    assert summary["max_duration"] > 0
    assert summary["average_throughput"] > 0

if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--benchmark-only"])

"""
Advanced benchmarking system for HeyGen AI tests.
Comprehensive performance testing and optimization.
"""

import pytest
import time
import asyncio
import statistics
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import random
import concurrent.futures
import threading
from contextlib import contextmanager

class BenchmarkType(Enum):
    """Benchmark types."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED = "mixed"

class BenchmarkScale(Enum):
    """Benchmark scale levels."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    name: str
    benchmark_type: BenchmarkType
    scale: BenchmarkScale
    iterations: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    median_duration: float
    std_deviation: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    warmup_iterations: int = 5
    measurement_iterations: int = 100
    timeout: float = 300.0
    memory_threshold: float = 1000.0  # MB
    cpu_threshold: float = 80.0  # %
    enable_gc: bool = True
    parallel_execution: bool = False
    max_workers: int = 4
    detailed_metrics: bool = True

class AdvancedBenchmark:
    """Advanced benchmarking system."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()
        self.baseline_metrics = self._get_baseline_metrics()
    
    def _get_baseline_metrics(self) -> Dict[str, float]:
        """Get baseline system metrics."""
        return {
            "baseline_memory": self.process.memory_info().rss / 1024 / 1024,
            "baseline_cpu": self.process.cpu_percent(),
            "baseline_timestamp": time.time()
        }
    
    def benchmark_function(self, 
                          name: str,
                          func: Callable,
                          benchmark_type: BenchmarkType = BenchmarkType.MIXED,
                          scale: BenchmarkScale = BenchmarkScale.MEDIUM,
                          *args, **kwargs) -> BenchmarkResult:
        """Benchmark a function with comprehensive metrics."""
        
        # Warmup phase
        if self.config.warmup_iterations > 0:
            for _ in range(self.config.warmup_iterations):
                try:
                    func(*args, **kwargs)
                except Exception:
                    pass  # Ignore warmup errors
        
        # Force garbage collection
        if self.config.enable_gc:
            gc.collect()
        
        # Measurement phase
        durations = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        for i in range(self.config.measurement_iterations):
            iteration_start = time.time()
            iteration_memory_start = self.process.memory_info().rss / 1024 / 1024
            iteration_cpu_start = self.process.cpu_percent()
            
            try:
                result = func(*args, **kwargs)
                success_count += 1
            except Exception as e:
                # Log error but continue benchmarking
                print(f"Benchmark iteration {i} failed: {e}")
            
            iteration_end = time.time()
            iteration_memory_end = self.process.memory_info().rss / 1024 / 1024
            iteration_cpu_end = self.process.cpu_percent()
            
            duration = iteration_end - iteration_start
            durations.append(duration)
            
            memory_usage.append(iteration_memory_end - iteration_memory_start)
            cpu_usage.append((iteration_cpu_start + iteration_cpu_end) / 2)
        
        end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        total_duration = end_time - start_time
        avg_duration = statistics.mean(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        median_duration = statistics.median(durations) if durations else 0
        std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0
        throughput = self.config.measurement_iterations / total_duration if total_duration > 0 else 0
        success_rate = success_count / self.config.measurement_iterations * 100
        
        # Memory and CPU metrics
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        avg_cpu_usage = statistics.mean(cpu_usage) if cpu_usage else 0
        
        # Create result
        result = BenchmarkResult(
            name=name,
            benchmark_type=benchmark_type,
            scale=scale,
            iterations=self.config.measurement_iterations,
            total_duration=total_duration,
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            median_duration=median_duration,
            std_deviation=std_deviation,
            throughput=throughput,
            memory_usage=avg_memory_usage,
            cpu_usage=avg_cpu_usage,
            success_rate=success_rate,
            metadata={
                "warmup_iterations": self.config.warmup_iterations,
                "baseline_memory": self.baseline_metrics["baseline_memory"],
                "final_memory": final_memory,
                "memory_delta": final_memory - initial_memory
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_async_function(self,
                                name: str,
                                async_func: Callable,
                                benchmark_type: BenchmarkType = BenchmarkType.MIXED,
                                scale: BenchmarkScale = BenchmarkScale.MEDIUM,
                                *args, **kwargs) -> BenchmarkResult:
        """Benchmark an async function."""
        
        def sync_wrapper():
            return asyncio.run(async_func(*args, **kwargs))
        
        return self.benchmark_function(name, sync_wrapper, benchmark_type, scale)
    
    def benchmark_parallel(self,
                          name: str,
                          func: Callable,
                          benchmark_type: BenchmarkType = BenchmarkType.MIXED,
                          scale: BenchmarkScale = BenchmarkScale.MEDIUM,
                          *args, **kwargs) -> BenchmarkResult:
        """Benchmark function with parallel execution."""
        
        def parallel_wrapper():
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(func, *args, **kwargs)
                    for _ in range(self.config.max_workers)
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            return results
        
        return self.benchmark_function(name, parallel_wrapper, benchmark_type, scale)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {"total_benchmarks": 0}
        
        total_benchmarks = len(self.results)
        total_duration = sum(r.total_duration for r in self.results)
        avg_duration = sum(r.avg_duration for r in self.results) / total_benchmarks
        avg_throughput = sum(r.throughput for r in self.results) / total_benchmarks
        avg_memory = sum(r.memory_usage for r in self.results) / total_benchmarks
        avg_cpu = sum(r.cpu_usage for r in self.results) / total_benchmarks
        avg_success_rate = sum(r.success_rate for r in self.results) / total_benchmarks
        
        return {
            "total_benchmarks": total_benchmarks,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "average_throughput": avg_throughput,
            "average_memory_usage": avg_memory,
            "average_cpu_usage": avg_cpu,
            "average_success_rate": avg_success_rate,
            "benchmarks": [
                {
                    "name": r.name,
                    "type": r.benchmark_type.value,
                    "scale": r.scale.value,
                    "avg_duration": r.avg_duration,
                    "throughput": r.throughput,
                    "success_rate": r.success_rate
                }
                for r in self.results
            ]
        }
    
    def save_results(self, file_path: str):
        """Save benchmark results to file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "measurement_iterations": self.config.measurement_iterations,
                "timeout": self.config.timeout,
                "parallel_execution": self.config.parallel_execution,
                "max_workers": self.config.max_workers
            },
            "baseline_metrics": self.baseline_metrics,
            "results": [
                {
                    "name": r.name,
                    "benchmark_type": r.benchmark_type.value,
                    "scale": r.scale.value,
                    "iterations": r.iterations,
                    "total_duration": r.total_duration,
                    "avg_duration": r.avg_duration,
                    "min_duration": r.min_duration,
                    "max_duration": r.max_duration,
                    "median_duration": r.median_duration,
                    "std_deviation": r.std_deviation,
                    "throughput": r.throughput,
                    "memory_usage": r.memory_usage,
                    "cpu_usage": r.cpu_usage,
                    "success_rate": r.success_rate,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in self.results
            ],
            "summary": self.get_summary()
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2)

class TestBenchmarks:
    """Benchmark tests."""
    
    def __init__(self):
        self.benchmark = AdvancedBenchmark()
    
    def test_cpu_intensive_benchmark(self):
        """Test CPU-intensive operations."""
        def cpu_intensive_task(n: int = 1000000):
            return sum(i * i for i in range(n))
        
        result = self.benchmark.benchmark_function(
            "cpu_intensive_task",
            cpu_intensive_task,
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.MEDIUM
        )
        
        assert result.benchmark_type == BenchmarkType.CPU_INTENSIVE
        assert result.iterations > 0
        assert result.avg_duration > 0
        assert result.throughput > 0
        assert result.success_rate > 90  # Should have high success rate
    
    def test_memory_intensive_benchmark(self):
        """Test memory-intensive operations."""
        def memory_intensive_task(size: int = 1000000):
            data = [random.random() for _ in range(size)]
            sorted_data = sorted(data)
            return len(sorted_data)
        
        result = self.benchmark.benchmark_function(
            "memory_intensive_task",
            memory_intensive_task,
            BenchmarkType.MEMORY_INTENSIVE,
            BenchmarkScale.MEDIUM
        )
        
        assert result.benchmark_type == BenchmarkType.MEMORY_INTENSIVE
        assert result.memory_usage > 0
        assert result.success_rate > 90
    
    def test_io_intensive_benchmark(self):
        """Test IO-intensive operations."""
        def io_intensive_task():
            # Simulate file operations
            temp_file = Path("temp_benchmark.txt")
            data = "test data " * 1000
            
            # Write
            temp_file.write_text(data)
            
            # Read
            content = temp_file.read_text()
            
            # Cleanup
            temp_file.unlink()
            
            return len(content)
        
        result = self.benchmark.benchmark_function(
            "io_intensive_task",
            io_intensive_task,
            BenchmarkType.IO_INTENSIVE,
            BenchmarkScale.SMALL
        )
        
        assert result.benchmark_type == BenchmarkType.IO_INTENSIVE
        assert result.success_rate > 90
    
    def test_async_benchmark(self):
        """Test async function benchmarking."""
        async def async_task(delay: float = 0.01):
            await asyncio.sleep(delay)
            return "async_result"
        
        result = self.benchmark.benchmark_async_function(
            "async_task",
            async_task,
            BenchmarkType.MIXED,
            BenchmarkScale.SMALL
        )
        
        assert result.avg_duration > 0
        assert result.success_rate > 90
    
    def test_parallel_benchmark(self):
        """Test parallel execution benchmarking."""
        def parallel_task(worker_id: int):
            # Simulate some work
            time.sleep(0.01)
            return f"worker_{worker_id}_result"
        
        result = self.benchmark.benchmark_parallel(
            "parallel_task",
            parallel_task,
            BenchmarkType.MIXED,
            BenchmarkScale.MEDIUM
        )
        
        assert result.avg_duration > 0
        assert result.success_rate > 90
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        def fast_function():
            return sum(range(1000))
        
        def slow_function():
            time.sleep(0.001)
            return sum(range(1000))
        
        fast_result = self.benchmark.benchmark_function(
            "fast_function",
            fast_function,
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.SMALL
        )
        
        slow_result = self.benchmark.benchmark_function(
            "slow_function",
            slow_function,
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.SMALL
        )
        
        # Fast function should be faster
        assert fast_result.avg_duration < slow_result.avg_duration
        assert fast_result.throughput > slow_result.throughput
    
    def test_benchmark_metadata(self):
        """Test benchmark metadata collection."""
        def metadata_task():
            return {"result": "test", "timestamp": datetime.now().isoformat()}
        
        result = self.benchmark.benchmark_function(
            "metadata_task",
            metadata_task,
            BenchmarkType.MIXED,
            BenchmarkScale.SMALL
        )
        
        assert "warmup_iterations" in result.metadata
        assert "baseline_memory" in result.metadata
        assert "final_memory" in result.metadata
        assert "memory_delta" in result.metadata
    
    def test_benchmark_summary(self):
        """Test benchmark summary generation."""
        # Run a few benchmarks
        self.benchmark.benchmark_function(
            "test1",
            lambda: sum(range(1000)),
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.SMALL
        )
        
        self.benchmark.benchmark_function(
            "test2",
            lambda: [i for i in range(1000)],
            BenchmarkType.MEMORY_INTENSIVE,
            BenchmarkScale.SMALL
        )
        
        summary = self.benchmark.get_summary()
        
        assert summary["total_benchmarks"] == 2
        assert "average_duration" in summary
        assert "average_throughput" in summary
        assert "average_memory_usage" in summary
        assert "average_cpu_usage" in summary
        assert "average_success_rate" in summary
        assert len(summary["benchmarks"]) == 2
    
    def test_benchmark_save_load(self):
        """Test benchmark results saving and loading."""
        # Run a benchmark
        self.benchmark.benchmark_function(
            "save_test",
            lambda: sum(range(1000)),
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.SMALL
        )
        
        # Save results
        report_file = Path("test_benchmark_report.json")
        self.benchmark.save_results(str(report_file))
        
        # Verify file exists and is valid JSON
        assert report_file.exists()
        
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        assert "timestamp" in data
        assert "config" in data
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "save_test"
        
        # Cleanup
        report_file.unlink()

class TestPerformanceOptimization:
    """Performance optimization tests."""
    
    def test_algorithm_comparison(self):
        """Test different algorithm implementations."""
        def bubble_sort(data):
            data = data.copy()
            n = len(data)
            for i in range(n):
                for j in range(0, n - i - 1):
                    if data[j] > data[j + 1]:
                        data[j], data[j + 1] = data[j + 1], data[j]
            return data
        
        def quick_sort(data):
            if len(data) <= 1:
                return data
            pivot = data[len(data) // 2]
            left = [x for x in data if x < pivot]
            middle = [x for x in data if x == pivot]
            right = [x for x in data if x > pivot]
            return quick_sort(left) + middle + quick_sort(right)
        
        # Generate test data
        test_data = [random.randint(1, 1000) for _ in range(1000)]
        
        benchmark = AdvancedBenchmark()
        
        # Benchmark bubble sort
        bubble_result = benchmark.benchmark_function(
            "bubble_sort",
            bubble_sort,
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.MEDIUM,
            test_data
        )
        
        # Benchmark quick sort
        quick_result = benchmark.benchmark_function(
            "quick_sort",
            quick_sort,
            BenchmarkType.CPU_INTENSIVE,
            BenchmarkScale.MEDIUM,
            test_data
        )
        
        # Quick sort should be faster
        assert quick_result.avg_duration < bubble_result.avg_duration
        assert quick_result.throughput > bubble_result.throughput
    
    def test_memory_optimization(self):
        """Test memory optimization techniques."""
        def memory_inefficient():
            # Create large lists and don't clean up
            data = []
            for i in range(10000):
                data.append([j for j in range(100)])
            return len(data)
        
        def memory_efficient():
            # Use generators and clean up
            total = 0
            for i in range(10000):
                data = [j for j in range(100)]
                total += len(data)
                del data  # Explicit cleanup
            return total
        
        benchmark = AdvancedBenchmark()
        
        # Benchmark memory inefficient version
        inefficient_result = benchmark.benchmark_function(
            "memory_inefficient",
            memory_inefficient,
            BenchmarkType.MEMORY_INTENSIVE,
            BenchmarkScale.MEDIUM
        )
        
        # Benchmark memory efficient version
        efficient_result = benchmark.benchmark_function(
            "memory_efficient",
            memory_efficient,
            BenchmarkType.MEMORY_INTENSIVE,
            BenchmarkScale.MEDIUM
        )
        
        # Efficient version should use less memory
        assert efficient_result.memory_usage < inefficient_result.memory_usage

# Test fixtures
@pytest.fixture
def benchmark_config():
    """Benchmark configuration fixture."""
    return BenchmarkConfig(
        warmup_iterations=2,
        measurement_iterations=10,
        timeout=60.0
    )

@pytest.fixture
def advanced_benchmark(benchmark_config):
    """Advanced benchmark fixture."""
    return AdvancedBenchmark(benchmark_config)

# Test markers
pytestmark = pytest.mark.usefixtures("advanced_benchmark")

if __name__ == "__main__":
    # Run the benchmark tests
    pytest.main([__file__, "-v"])

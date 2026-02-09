"""
Performance Benchmark for Color Grading AI
===========================================

Performance benchmarking and comparison tool.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    throughput: float  # operations per second
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison result."""
    benchmark_a: BenchmarkResult
    benchmark_b: BenchmarkResult
    speedup: float
    improvement_percent: float
    winner: str


class PerformanceBenchmark:
    """
    Performance benchmark tool.
    
    Features:
    - Function benchmarking
    - Multiple iterations
    - Statistical analysis
    - Comparison between implementations
    - Throughput calculation
    - Memory profiling (optional)
    """
    
    def __init__(self):
        """Initialize performance benchmark."""
        self._results: List[BenchmarkResult] = []
    
    async def benchmark(
        self,
        name: str,
        function: Callable,
        iterations: int = 100,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a function.
        
        Args:
            name: Benchmark name
            function: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark result
        """
        # Warmup
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                function(*args, **kwargs)
        
        # Benchmark
        times = []
        start_total = time.time()
        
        for _ in range(iterations):
            start = time.time()
            
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                function(*args, **kwargs)
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        total_time = time.time() - start_total
        
        # Calculate statistics
        avg_time = mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = median(times)
        std_dev = stdev(times) if len(times) > 1 else 0.0
        throughput = iterations / total_time if total_time > 0 else 0.0
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput
        )
        
        self._results.append(result)
        logger.info(f"Benchmark completed: {name} - {avg_time*1000:.2f}ms avg")
        
        return result
    
    def compare(
        self,
        benchmark_a: BenchmarkResult,
        benchmark_b: BenchmarkResult
    ) -> ComparisonResult:
        """
        Compare two benchmark results.
        
        Args:
            benchmark_a: First benchmark result
            benchmark_b: Second benchmark result
            
        Returns:
            Comparison result
        """
        speedup = benchmark_a.avg_time / benchmark_b.avg_time if benchmark_b.avg_time > 0 else 0.0
        improvement = ((benchmark_a.avg_time - benchmark_b.avg_time) / benchmark_a.avg_time * 100) if benchmark_a.avg_time > 0 else 0.0
        winner = benchmark_b.name if benchmark_b.avg_time < benchmark_a.avg_time else benchmark_a.name
        
        return ComparisonResult(
            benchmark_a=benchmark_a,
            benchmark_b=benchmark_b,
            speedup=speedup,
            improvement_percent=improvement,
            winner=winner
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        if not self._results:
            return {
                "total_benchmarks": 0,
            }
        
        return {
            "total_benchmarks": len(self._results),
            "fastest": min(self._results, key=lambda x: x.avg_time).name,
            "slowest": max(self._results, key=lambda x: x.avg_time).name,
            "avg_throughput": mean(r.throughput for r in self._results),
        }



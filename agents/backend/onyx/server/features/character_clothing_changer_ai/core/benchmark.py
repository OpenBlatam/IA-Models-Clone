"""
Benchmark System
================

System for benchmarking operations and performance testing.
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime

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
    p50: float
    p95: float
    p99: float
    success_count: int
    error_count: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "median_time": self.median_time,
            "std_dev": self.std_dev,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "errors": self.errors,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class BenchmarkRunner:
    """Benchmark runner for operations."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.results: Dict[str, BenchmarkResult] = {}
    
    async def benchmark(
        self,
        name: str,
        func: Callable[[], Awaitable[Any]],
        iterations: int = 10,
        warmup: int = 2,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark an async function.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            metadata: Optional metadata
            
        Returns:
            Benchmark result
        """
        logger.info(f"Starting benchmark: {name} ({iterations} iterations)")
        
        # Warmup
        for _ in range(warmup):
            try:
                await func()
            except Exception:
                pass
        
        # Benchmark
        times = []
        errors = []
        success_count = 0
        error_count = 0
        
        for i in range(iterations):
            start = time.time()
            try:
                await func()
                elapsed = time.time() - start
                times.append(elapsed)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(str(e))
                logger.warning(f"Benchmark iteration {i+1} failed: {e}")
        
        if not times:
            raise ValueError(f"All benchmark iterations failed for {name}")
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        # Percentiles
        sorted_times = sorted(times)
        p50 = sorted_times[int(len(sorted_times) * 0.50)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0]
        p99 = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0]
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            p50=p50,
            p95=p95,
            p99=p99,
            success_count=success_count,
            error_count=error_count,
            errors=errors[:10],  # Limit errors
            metadata=metadata or {}
        )
        
        self.results[name] = result
        logger.info(f"Benchmark {name} completed: avg={avg_time:.3f}s, p95={p95:.3f}s")
        
        return result
    
    def compare(self, *names: str) -> Dict[str, Any]:
        """
        Compare multiple benchmarks.
        
        Args:
            *names: Benchmark names to compare
            
        Returns:
            Comparison dictionary
        """
        results = [self.results[name] for name in names if name in self.results]
        
        if not results:
            return {}
        
        comparison = {
            "benchmarks": [r.name for r in results],
            "avg_times": [r.avg_time for r in results],
            "p95_times": [r.p95 for r in results],
            "p99_times": [r.p99 for r in results],
            "success_rates": [r.success_count / r.iterations for r in results]
        }
        
        # Find fastest
        fastest = min(results, key=lambda r: r.avg_time)
        comparison["fastest"] = fastest.name
        
        return comparison
    
    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get benchmark result by name."""
        return self.results.get(name)
    
    def get_all_results(self) -> Dict[str, BenchmarkResult]:
        """Get all benchmark results."""
        return self.results.copy()
    
    def clear_results(self):
        """Clear all results."""
        self.results.clear()


class PerformanceProfiler:
    """Performance profiler for detailed analysis."""
    
    def __init__(self):
        """Initialize profiler."""
        from collections import defaultdict
        self.profiles: Dict[str, List[float]] = defaultdict(list)
    
    def start(self, operation: str) -> float:
        """
        Start profiling an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Start time
        """
        return time.time()
    
    def end(self, operation: str, start_time: float):
        """
        End profiling an operation.
        
        Args:
            operation: Operation name
            start_time: Start time from start()
        """
        elapsed = time.time() - start_time
        self.profiles[operation].append(elapsed)
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Statistics dictionary
        """
        if operation not in self.profiles or not self.profiles[operation]:
            return None
        
        times = self.profiles[operation]
        
        return {
            "count": len(times),
            "total": sum(times),
            "avg": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "median": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            op: self.get_stats(op)
            for op in self.profiles.keys()
        }
    
    def clear(self, operation: Optional[str] = None):
        """
        Clear profiles.
        
        Args:
            operation: Optional operation name (clears all if not provided)
        """
        if operation:
            self.profiles.pop(operation, None)
        else:
            self.profiles.clear()


"""
Benchmark Runner
================

Performance benchmark runner.
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
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
    p95_time: float
    p99_time: float
    throughput: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BenchmarkRunner:
    """Performance benchmark runner."""
    
    def __init__(self):
        self._results: Dict[str, BenchmarkResult] = {}
    
    async def benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Run benchmark."""
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                continue
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        if not times:
            raise ValueError("No successful iterations")
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        
        sorted_times = sorted(times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)
        p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        p99_time = sorted_times[min(p99_index, len(sorted_times) - 1)]
        
        throughput = iterations / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput=throughput
        )
        
        self._results[name] = result
        logger.info(f"Benchmark {name} completed: {avg_time:.4f}s avg, {throughput:.2f} ops/s")
        
        return result
    
    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get benchmark result."""
        return self._results.get(name)
    
    def get_all_results(self) -> Dict[str, BenchmarkResult]:
        """Get all benchmark results."""
        return self._results.copy()
    
    def compare_results(self, name1: str, name2: str) -> Dict[str, Any]:
        """Compare two benchmark results."""
        result1 = self.get_result(name1)
        result2 = self.get_result(name2)
        
        if not result1 or not result2:
            return {"error": "One or both results not found"}
        
        return {
            "name1": name1,
            "name2": name2,
            "avg_time_diff": result2.avg_time - result1.avg_time,
            "avg_time_diff_percent": ((result2.avg_time - result1.avg_time) / result1.avg_time * 100) if result1.avg_time > 0 else 0,
            "throughput_diff": result2.throughput - result1.throughput,
            "throughput_diff_percent": ((result2.throughput - result1.throughput) / result1.throughput * 100) if result1.throughput > 0 else 0
        }


# Import asyncio
import asyncio
















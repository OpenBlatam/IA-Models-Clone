"""
Performance Benchmark - Benchmark de Performance
===============================================

Benchmarking de performance:
- Function benchmarking
- API endpoint benchmarking
- Database query benchmarking
- Memory profiling
- CPU profiling
"""

import logging
import time
import statistics
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Resultado de benchmark"""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.runs: List[float] = []
        self.timestamps: List[datetime] = []
    
    def add_run(self, duration: float) -> None:
        """Agrega resultado de ejecución"""
        self.runs.append(duration)
        self.timestamps.append(datetime.now())
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        if not self.runs:
            return {}
        
        return {
            "name": self.name,
            "runs": len(self.runs),
            "min": min(self.runs),
            "max": max(self.runs),
            "mean": statistics.mean(self.runs),
            "median": statistics.median(self.runs),
            "stdev": statistics.stdev(self.runs) if len(self.runs) > 1 else 0,
            "p95": self._percentile(95),
            "p99": self._percentile(99)
        }
    
    def _percentile(self, p: float) -> float:
        """Calcula percentil"""
        if not self.runs:
            return 0.0
        sorted_runs = sorted(self.runs)
        index = int(len(sorted_runs) * p / 100)
        return sorted_runs[min(index, len(sorted_runs) - 1)]


class PerformanceBenchmark:
    """
    Benchmark de performance.
    """
    
    def __init__(self) -> None:
        self.benchmarks: Dict[str, BenchmarkResult] = {}
    
    def benchmark(
        self,
        name: Optional[str] = None,
        iterations: int = 100
    ):
        """Decorator para benchmarking"""
        def decorator(func: Callable) -> Callable:
            benchmark_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = None
                for _ in range(iterations):
                    start = time.perf_counter()
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    self._record_benchmark(benchmark_name, duration)
                return result
            
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = None
                for _ in range(iterations):
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    self._record_benchmark(benchmark_name, duration)
                return result
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def _record_benchmark(self, name: str, duration: float) -> None:
        """Registra resultado de benchmark"""
        if name not in self.benchmarks:
            self.benchmarks[name] = BenchmarkResult(name)
        self.benchmarks[name].add_run(duration)
    
    async def run_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        *args: Any,
        **kwargs: Any
    ) -> BenchmarkResult:
        """Ejecuta benchmark"""
        benchmark = BenchmarkResult(name)
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
            
            duration = time.perf_counter() - start
            benchmark.add_run(duration)
        
        self.benchmarks[name] = benchmark
        return benchmark
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene todas las estadísticas"""
        return {
            name: benchmark.get_stats()
            for name, benchmark in self.benchmarks.items()
        }
    
    def compare_benchmarks(self, names: List[str]) -> Dict[str, Any]:
        """Compara múltiples benchmarks"""
        stats = {}
        for name in names:
            if name in self.benchmarks:
                stats[name] = self.benchmarks[name].get_stats()
        
        return {
            "comparison": stats,
            "fastest": min(
                stats.items(),
                key=lambda x: x[1].get("mean", float("inf"))
            )[0] if stats else None
        }


import asyncio


def get_performance_benchmark() -> PerformanceBenchmark:
    """Obtiene benchmark de performance"""
    return PerformanceBenchmark()
















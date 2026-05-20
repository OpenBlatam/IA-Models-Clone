"""
Benchmark Module

Performance benchmarking and comparison utilities.
"""

from typing import Dict, Any, List, Optional, Callable
import time
import logging
from dataclasses import dataclass
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    success: bool
    error: Optional[str] = None


class BenchmarkRunner:
    """
    Runs benchmarks on generator operations.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a function.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            iterations: Number of iterations
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            BenchmarkResult
        """
        times = []
        success = True
        error = None
        
        logger.info(f"Running benchmark: {name} ({iterations} iterations)")
        
        for i in range(iterations):
            try:
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                success = False
                error = str(e)
                logger.error(f"Benchmark {name} failed: {e}")
                break
        
        if not times:
            return BenchmarkResult(
                name=name,
                iterations=0,
                total_time=0.0,
                avg_time=0.0,
                min_time=0.0,
                max_time=0.0,
                median_time=0.0,
                std_dev=0.0,
                success=False,
                error=error
            )
        
        result = BenchmarkResult(
            name=name,
            iterations=len(times),
            total_time=sum(times),
            avg_time=mean(times),
            min_time=min(times),
            max_time=max(times),
            median_time=median(times),
            std_dev=stdev(times) if len(times) > 1 else 0.0,
            success=success,
            error=error
        )
        
        self.results.append(result)
        return result
    
    def compare_configs(
        self,
        configs: List[Dict[str, Any]],
        create_func: Callable,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Compare multiple configurations.
        
        Args:
            configs: List of configurations to compare
            create_func: Function to create generator
            iterations: Number of iterations per config
            
        Returns:
            Comparison results
        """
        comparison = {
            "configs": [],
            "results": []
        }
        
        for i, config in enumerate(configs):
            config_name = config.get("name", f"config_{i}")
            comparison["configs"].append(config_name)
            
            result = self.benchmark(
                f"config_{i}",
                create_func,
                iterations=iterations,
                **config
            )
            
            comparison["results"].append({
                "name": config_name,
                "avg_time": result.avg_time,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "success": result.success
            })
        
        # Find fastest
        successful = [r for r in comparison["results"] if r["success"]]
        if successful:
            fastest = min(successful, key=lambda x: x["avg_time"])
            comparison["fastest"] = fastest["name"]
            comparison["fastest_time"] = fastest["avg_time"]
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        return {
            "total_benchmarks": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "results": [
                {
                    "name": r.name,
                    "avg_time": r.avg_time,
                    "iterations": r.iterations,
                    "success": r.success
                }
                for r in self.results
            ]
        }
    
    def reset(self) -> None:
        """Reset benchmark results."""
        self.results.clear()


def create_benchmark_runner() -> BenchmarkRunner:
    """Create a new benchmark runner."""
    return BenchmarkRunner()
















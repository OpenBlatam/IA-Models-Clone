"""
Benchmarking System
===================

Sistema de benchmarking y comparación de performance.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de benchmark."""
    name: str
    iterations: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float
    throughput: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Ejecutor de benchmarks.
    
    Ejecuta benchmarks y compara performance.
    """
    
    def __init__(self):
        """Inicializar ejecutor de benchmarks."""
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Ejecutar benchmark.
        
        Args:
            name: Nombre del benchmark
            func: Función a benchmarkear
            iterations: Número de iteraciones
            warmup_iterations: Iteraciones de calentamiento
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado del benchmark
        """
        logger.info(f"Running benchmark: {name} ({iterations} iterations)")
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in warmup: {e}")
        
        # Ejecutar benchmark
        times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                errors += 1
                logger.warning(f"Error in iteration {i}: {e}")
        
        if not times:
            raise RuntimeError("All benchmark iterations failed")
        
        # Calcular estadísticas
        times_sorted = sorted(times)
        total_time = sum(times)
        average_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        p95_time = times_sorted[int(len(times_sorted) * 0.95)]
        p99_time = times_sorted[int(len(times_sorted) * 0.99)]
        throughput = len(times) / total_time if total_time > 0 else 0.0
        
        result = BenchmarkResult(
            name=name,
            iterations=len(times),
            total_time=total_time,
            average_time=average_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput=throughput,
            metadata={
                "errors": errors,
                "error_rate": errors / iterations if iterations > 0 else 0.0
            }
        )
        
        self.results.append(result)
        logger.info(f"Benchmark completed: {name} - avg: {average_time:.4f}s, throughput: {throughput:.2f} ops/s")
        
        return result
    
    def compare_benchmarks(
        self,
        benchmark_names: List[str]
    ) -> Dict[str, Any]:
        """
        Comparar benchmarks.
        
        Args:
            benchmark_names: Nombres de benchmarks a comparar
            
        Returns:
            Comparación de benchmarks
        """
        benchmarks = [
            r for r in self.results
            if r.name in benchmark_names
        ]
        
        if not benchmarks:
            return {"error": "No benchmarks found"}
        
        if len(benchmarks) < 2:
            return {"error": "Need at least 2 benchmarks to compare"}
        
        # Encontrar el más rápido
        fastest = min(benchmarks, key=lambda x: x.average_time)
        
        comparison = {
            "benchmarks": [
                {
                    "name": b.name,
                    "average_time": b.average_time,
                    "throughput": b.throughput,
                    "p95_time": b.p95_time
                }
                for b in benchmarks
            ],
            "fastest": fastest.name,
            "comparison": {}
        }
        
        # Comparar con el más rápido
        for benchmark in benchmarks:
            if benchmark.name != fastest.name:
                speedup = fastest.average_time / benchmark.average_time
                comparison["comparison"][benchmark.name] = {
                    "speedup": speedup,
                    "slower_by": (benchmark.average_time - fastest.average_time) / fastest.average_time * 100
                }
        
        return comparison
    
    def get_all_results(self) -> List[BenchmarkResult]:
        """Obtener todos los resultados."""
        return self.results
    
    def clear_results(self) -> None:
        """Limpiar resultados."""
        self.results = []
        logger.info("Benchmark results cleared")


# Instancia global
_benchmark_runner: Optional[BenchmarkRunner] = None


def get_benchmark_runner() -> BenchmarkRunner:
    """Obtener instancia global del ejecutor de benchmarks."""
    global _benchmark_runner
    if _benchmark_runner is None:
        _benchmark_runner = BenchmarkRunner()
    return _benchmark_runner







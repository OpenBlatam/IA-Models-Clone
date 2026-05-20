"""
Benchmarking - Sistema de Benchmarking
=======================================

Sistema de benchmarking y performance testing.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de benchmark."""
    benchmark_id: str
    name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    average_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float
    total_time: float
    started_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Ejecutor de benchmarks."""
    
    def __init__(self):
        self.results: Dict[str, BenchmarkResult] = {}
    
    async def run_benchmark(
        self,
        benchmark_id: str,
        name: str,
        function: Callable,
        iterations: int = 10,
        warmup_runs: int = 2,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Ejecutar benchmark.
        
        Args:
            benchmark_id: ID único del benchmark
            name: Nombre del benchmark
            function: Función a ejecutar
            iterations: Número de iteraciones
            warmup_runs: Ejecuciones de calentamiento
            **kwargs: Argumentos para la función
        
        Returns:
            Resultado del benchmark
        """
        logger.info(f"Running benchmark: {name} ({iterations} iterations)")
        
        started_at = datetime.now()
        times = []
        successful_runs = 0
        failed_runs = 0
        
        # Warmup runs
        if warmup_runs > 0:
            logger.debug(f"Warmup runs: {warmup_runs}")
            for _ in range(warmup_runs):
                try:
                    if asyncio.iscoroutinefunction(function):
                        await function(**kwargs)
                    else:
                        function(**kwargs)
                except Exception:
                    pass  # Ignorar errores en warmup
        
        # Benchmark runs
        for i in range(iterations):
            try:
                start_time = time.perf_counter()
                
                if asyncio.iscoroutinefunction(function):
                    await function(**kwargs)
                else:
                    function(**kwargs)
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                
                times.append(elapsed)
                successful_runs += 1
                
            except Exception as e:
                failed_runs += 1
                logger.warning(f"Benchmark iteration {i+1} failed: {e}")
        
        completed_at = datetime.now()
        total_time = (completed_at - started_at).total_seconds()
        
        if not times:
            raise Exception("All benchmark iterations failed")
        
        # Calcular estadísticas
        times_sorted = sorted(times)
        average_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        
        # Percentiles
        p95_index = int(len(times_sorted) * 0.95)
        p95_time = times_sorted[p95_index] if p95_index < len(times_sorted) else times_sorted[-1]
        
        p99_index = int(len(times_sorted) * 0.99)
        p99_time = times_sorted[p99_index] if p99_index < len(times_sorted) else times_sorted[-1]
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            name=name,
            total_runs=iterations,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            average_time=average_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            total_time=total_time,
            started_at=started_at,
            completed_at=completed_at,
        )
        
        self.results[benchmark_id] = result
        
        logger.info(
            f"Benchmark completed: {name} - "
            f"avg: {average_time:.4f}s, "
            f"p95: {p95_time:.4f}s, "
            f"p99: {p99_time:.4f}s"
        )
        
        return result
    
    def get_result(self, benchmark_id: str) -> Optional[BenchmarkResult]:
        """Obtener resultado de benchmark."""
        return self.results.get(benchmark_id)
    
    def list_results(self) -> List[Dict[str, Any]]:
        """Listar todos los resultados."""
        return [
            {
                "benchmark_id": r.benchmark_id,
                "name": r.name,
                "average_time": r.average_time,
                "p95_time": r.p95_time,
                "p99_time": r.p99_time,
                "success_rate": r.successful_runs / r.total_runs if r.total_runs > 0 else 0.0,
            }
            for r in self.results.values()
        ]

































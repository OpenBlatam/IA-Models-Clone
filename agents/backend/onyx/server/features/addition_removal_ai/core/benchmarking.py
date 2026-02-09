"""
Benchmarking - Sistema de benchmarking comparativo
"""

import logging
import time
import asyncio
import statistics
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de benchmark"""
    name: str
    duration: float
    iterations: int
    ops_per_second: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BenchmarkSuite:
    """Suite de benchmarks"""

    def __init__(self):
        """Inicializar suite de benchmarks"""
        self.results: List[BenchmarkResult] = []

    async def benchmark_function(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Hacer benchmark de una función.

        Args:
            name: Nombre del benchmark
            func: Función a medir
            iterations: Número de iteraciones
            warmup: Iteraciones de calentamiento
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre

        Returns:
            Resultado del benchmark
        """
        # Warmup
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        
        # Benchmark real
        durations = []
        for _ in range(iterations):
            start = time.perf_counter()
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
            duration = time.perf_counter() - start
            durations.append(duration)
        
        total_time = sum(durations)
        avg_time = total_time / iterations
        ops_per_second = iterations / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            duration=total_time,
            iterations=iterations,
            ops_per_second=ops_per_second,
            metadata={
                "avg_time": avg_time,
                "min_time": min(durations),
                "max_time": max(durations),
                "median_time": sorted(durations)[len(durations) // 2]
            }
        )
        
        self.results.append(result)
        logger.info(f"Benchmark completado: {name} - {ops_per_second:.2f} ops/s")
        
        return result

    async def compare_implementations(
        self,
        implementations: List[Dict[str, Any]],
        test_data: Any,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Comparar diferentes implementaciones.

        Args:
            implementations: Lista de implementaciones a comparar
            test_data: Datos de prueba
            iterations: Número de iteraciones

        Returns:
            Comparación de resultados
        """
        comparison = {
            "implementations": [],
            "winner": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        best_ops = 0
        
        for impl in implementations:
            result = await self.benchmark_function(
                impl["name"],
                impl["func"],
                iterations,
                test_data,
                *impl.get("args", []),
                **impl.get("kwargs", {})
            )
            
            comparison["implementations"].append({
                "name": impl["name"],
                "ops_per_second": result.ops_per_second,
                "avg_time": result.metadata["avg_time"],
                "duration": result.duration
            })
            
            if result.ops_per_second > best_ops:
                best_ops = result.ops_per_second
                comparison["winner"] = impl["name"]
        
        return comparison

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de resultados.

        Returns:
            Resumen de benchmarks
        """
        if not self.results:
            return {"total_benchmarks": 0}
        
        return {
            "total_benchmarks": len(self.results),
            "avg_ops_per_second": statistics.mean([r.ops_per_second for r in self.results]),
            "best_performer": max(self.results, key=lambda r: r.ops_per_second).name,
            "worst_performer": min(self.results, key=lambda r: r.ops_per_second).name
        }


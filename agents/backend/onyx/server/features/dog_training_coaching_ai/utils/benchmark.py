"""
Benchmark Utilities
===================
Utilidades para benchmarking y performance testing.
"""

import time
import asyncio
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime
from statistics import mean, median, stdev
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger(__name__)


class Benchmark:
    """Clase para benchmarking de funciones."""
    
    def __init__(self, name: str = "benchmark"):
        """
        Inicializar benchmark.
        
        Args:
            name: Nombre del benchmark
        """
        self.name = name
        self.results: List[float] = []
        self.metadata: Dict[str, Any] = {}
    
    @contextmanager
    def measure(self):
        """Context manager para medir tiempo de ejecución."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.results.append(elapsed)
    
    async def measure_async(self, coro: Callable):
        """
        Medir tiempo de coroutine.
        
        Args:
            coro: Coroutine a medir
            
        Returns:
            Resultado de la coroutine
        """
        start = time.perf_counter()
        try:
            result = await coro
            return result
        finally:
            elapsed = time.perf_counter() - start
            self.results.append(elapsed)
    
    def run(self, func: Callable, iterations: int = 1, *args, **kwargs) -> Dict[str, Any]:
        """
        Ejecutar función múltiples veces y medir.
        
        Args:
            func: Función a ejecutar
            iterations: Número de iteraciones
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Estadísticas del benchmark
        """
        self.results.clear()
        
        for _ in range(iterations):
            with self.measure():
                func(*args, **kwargs)
        
        return self.get_stats()
    
    async def run_async(
        self,
        coro: Callable,
        iterations: int = 1,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecutar coroutine múltiples veces y medir.
        
        Args:
            coro: Coroutine a ejecutar
            iterations: Número de iteraciones
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Estadísticas del benchmark
        """
        self.results.clear()
        
        for _ in range(iterations):
            await self.measure_async(coro(*args, **kwargs))
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del benchmark."""
        if not self.results:
            return {
                "name": self.name,
                "iterations": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "median_time": 0,
                "std_dev": 0
            }
        
        return {
            "name": self.name,
            "iterations": len(self.results),
            "total_time": sum(self.results),
            "avg_time": mean(self.results),
            "min_time": min(self.results),
            "max_time": max(self.results),
            "median_time": median(self.results),
            "std_dev": stdev(self.results) if len(self.results) > 1 else 0,
            "results": self.results.copy()
        }
    
    def compare(self, other: "Benchmark") -> Dict[str, Any]:
        """
        Comparar con otro benchmark.
        
        Args:
            other: Otro benchmark
            
        Returns:
            Comparación
        """
        stats1 = self.get_stats()
        stats2 = other.get_stats()
        
        return {
            "benchmark1": stats1,
            "benchmark2": stats2,
            "speedup": stats2["avg_time"] / stats1["avg_time"] if stats1["avg_time"] > 0 else 0,
            "improvement_percent": (
                (stats2["avg_time"] - stats1["avg_time"]) / stats1["avg_time"] * 100
                if stats1["avg_time"] > 0 else 0
            )
        }


def benchmark_function(
    func: Callable,
    iterations: int = 100,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark de función.
    
    Args:
        func: Función a benchmarkear
        iterations: Número de iteraciones
        *args: Argumentos posicionales
        **kwargs: Argumentos nombrados
        
    Returns:
        Estadísticas
    """
    benchmark = Benchmark(func.__name__)
    return benchmark.run(func, iterations, *args, **kwargs)


async def benchmark_async_function(
    coro: Callable,
    iterations: int = 100,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark de función async.
    
    Args:
        coro: Coroutine a benchmarkear
        iterations: Número de iteraciones
        *args: Argumentos posicionales
        **kwargs: Argumentos nombrados
        
    Returns:
        Estadísticas
    """
    benchmark = Benchmark(coro.__name__)
    return await benchmark.run_async(coro, iterations, *args, **kwargs)




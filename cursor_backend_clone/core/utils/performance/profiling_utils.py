"""
Profiling Utils - Utilidades de Profiling y Benchmarking
=========================================================

Utilidades para profiling, benchmarking y análisis de rendimiento.
"""

import logging
import time
import cProfile
import pstats
import io
from typing import Any, Callable, Optional, Dict, List
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de benchmark"""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    timestamp: datetime = field(default_factory=datetime.now)


def benchmark(
    func: Callable,
    iterations: int = 1000,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Ejecutar benchmark de función.
    
    Args:
        func: Función a benchmarkear
        iterations: Número de iteraciones
        warmup: Iteraciones de warmup
        
    Returns:
        BenchmarkResult
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=sum(times),
        avg_time=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times)
    )


async def benchmark_async(
    func: Callable,
    iterations: int = 1000,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Ejecutar benchmark de función async.
    
    Args:
        func: Función async a benchmarkear
        iterations: Número de iteraciones
        warmup: Iteraciones de warmup
        
    Returns:
        BenchmarkResult
    """
    import asyncio
    
    # Warmup
    for _ in range(warmup):
        await func()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        end = time.perf_counter()
        times.append(end - start)
    
    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=sum(times),
        avg_time=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times)
    )


@contextmanager
def profile_context(output_file: Optional[str] = None):
    """
    Context manager para profiling.
    
    Args:
        output_file: Archivo opcional para guardar resultados
        
    Yields:
        Profiler
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        if output_file:
            profiler.dump_stats(output_file)
        else:
            # Imprimir estadísticas
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
            logger.info(f"Profiling results:\n{s.getvalue()}")


def profile_function(func: Callable) -> Callable:
    """
    Decorador para perfilar función.
    
    Args:
        func: Función a perfilar
        
    Returns:
        Función decorada
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with profile_context():
            return func(*args, **kwargs)
    
    return wrapper


class PerformanceTracker:
    """
    Tracker de rendimiento para múltiples operaciones.
    """
    
    def __init__(self):
        self.operations: Dict[str, List[float]] = {}
    
    def record(self, operation: str, duration: float) -> None:
        """
        Registrar duración de operación.
        
        Args:
            operation: Nombre de operación
            duration: Duración en segundos
        """
        if operation not in self.operations:
            self.operations[operation] = []
        self.operations[operation].append(duration)
    
    @contextmanager
    def track(self, operation: str):
        """
        Context manager para trackear operación.
        
        Args:
            operation: Nombre de operación
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(operation, duration)
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """
        Obtener estadísticas de operación.
        
        Args:
            operation: Nombre de operación
            
        Returns:
            Diccionario con estadísticas o None
        """
        if operation not in self.operations:
            return None
        
        times = self.operations[operation]
        return {
            "count": len(times),
            "total": sum(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener estadísticas de todas las operaciones.
        
        Returns:
            Diccionario con estadísticas por operación
        """
        return {
            op: self.get_stats(op)
            for op in self.operations.keys()
        }
    
    def clear(self) -> None:
        """Limpiar todas las estadísticas"""
        self.operations.clear()


class MemoryProfiler:
    """
    Profiler de memoria.
    """
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """
        Obtener uso de memoria actual.
        
        Returns:
            Diccionario con información de memoria
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            
            return {
                "rss": mem_info.rss,  # Resident Set Size
                "vms": mem_info.vms,  # Virtual Memory Size
                "percent": process.memory_percent(),
                "available": psutil.virtual_memory().available
            }
        except ImportError:
            logger.warning("psutil not available for memory profiling")
            return {}
    
    @staticmethod
    @contextmanager
    def track_memory(operation: str = "operation"):
        """
        Context manager para trackear uso de memoria.
        
        Args:
            operation: Nombre de operación
        """
        before = MemoryProfiler.get_memory_usage()
        
        try:
            yield
        finally:
            after = MemoryProfiler.get_memory_usage()
            
            if before and after:
                rss_diff = after.get("rss", 0) - before.get("rss", 0)
                logger.debug(
                    f"Memory usage for {operation}: "
                    f"RSS diff: {rss_diff / 1024 / 1024:.2f} MB"
                )


def compare_functions(
    *functions: Callable,
    iterations: int = 1000,
    warmup: int = 10
) -> List[BenchmarkResult]:
    """
    Comparar rendimiento de múltiples funciones.
    
    Args:
        *functions: Funciones a comparar
        iterations: Número de iteraciones
        warmup: Iteraciones de warmup
        
    Returns:
        Lista de BenchmarkResult
    """
    results = []
    
    for func in functions:
        result = benchmark(func, iterations=iterations, warmup=warmup)
        results.append(result)
        logger.info(
            f"{result.name}: avg={result.avg_time*1000:.3f}ms, "
            f"min={result.min_time*1000:.3f}ms, max={result.max_time*1000:.3f}ms"
        )
    
    return results





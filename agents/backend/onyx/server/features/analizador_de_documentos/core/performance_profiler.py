"""
Profiler de Rendimiento Avanzado
==================================

Sistema para profiling y optimización de rendimiento.
"""

import logging
import time
import tracemalloc
import cProfile
import pstats
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento"""
    function_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: float
    memory_peak: float
    cpu_percent: float


class PerformanceProfiler:
    """
    Profiler de rendimiento
    
    Proporciona:
    - Profiling de funciones
    - Análisis de memoria
    - Análisis de CPU
    - Detección de cuellos de botella
    - Recomendaciones de optimización
    """
    
    def __init__(self):
        """Inicializar profiler"""
        self.profiles: Dict[str, cProfile.Profile] = {}
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
        logger.info("PerformanceProfiler inicializado")
    
    @contextmanager
    def profile_function(self, function_name: str):
        """Context manager para profiling"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        tracemalloc.start()
        
        start_time = time.time()
        start_memory = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = tracemalloc.take_snapshot()
            profiler.disable()
            
            duration = end_time - start_time
            
            # Calcular memoria
            memory_diff = end_memory.compare_to(start_memory, 'lineno')
            memory_usage = sum(stat.size_diff for stat in memory_diff)
            memory_peak = tracemalloc.get_traced_memory()[1]
            
            # Guardar métricas
            if function_name not in self.metrics:
                self.metrics[function_name] = []
            
            self.metrics[function_name].append(PerformanceMetrics(
                function_name=function_name,
                call_count=1,
                total_time=duration,
                avg_time=duration,
                min_time=duration,
                max_time=duration,
                memory_usage=memory_usage,
                memory_peak=memory_peak,
                cpu_percent=0.0  # Se calcularía con psutil
            ))
            
            tracemalloc.stop()
    
    def profile(self, function_name: Optional[str] = None):
        """
        Decorator para profiling
        
        Args:
            function_name: Nombre de la función (opcional)
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = function_name or func.__name__
                with self.profile_function(name):
                    return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = function_name or func.__name__
                with self.profile_function(name):
                    return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def get_statistics(
        self,
        function_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        if function_name:
            if function_name not in self.metrics:
                return {}
            
            metrics_list = self.metrics[function_name]
            if not metrics_list:
                return {}
            
            return {
                "function_name": function_name,
                "total_calls": len(metrics_list),
                "avg_time": sum(m.total_time for m in metrics_list) / len(metrics_list),
                "min_time": min(m.min_time for m in metrics_list),
                "max_time": max(m.max_time for m in metrics_list),
                "avg_memory": sum(m.memory_usage for m in metrics_list) / len(metrics_list),
                "peak_memory": max(m.memory_peak for m in metrics_list)
            }
        
        # Estadísticas globales
        all_stats = {}
        for func_name in self.metrics.keys():
            all_stats[func_name] = self.get_statistics(func_name)
        
        return all_stats
    
    def get_bottlenecks(
        self,
        threshold_ms: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Detectar cuellos de botella
        
        Args:
            threshold_ms: Umbral en milisegundos
        
        Returns:
            Lista de funciones que exceden el umbral
        """
        bottlenecks = []
        
        for function_name, stats in self.get_statistics().items():
            if stats.get("avg_time", 0) * 1000 > threshold_ms:
                bottlenecks.append({
                    "function": function_name,
                    "avg_time_ms": stats["avg_time"] * 1000,
                    "calls": stats["total_calls"],
                    "recommendation": self._generate_recommendation(stats)
                })
        
        return sorted(bottlenecks, key=lambda x: x["avg_time_ms"], reverse=True)
    
    def _generate_recommendation(
        self,
        stats: Dict[str, Any]
    ) -> str:
        """Generar recomendación de optimización"""
        recommendations = []
        
        if stats.get("avg_time", 0) > 5.0:
            recommendations.append("Considerar procesamiento asíncrono")
        
        if stats.get("peak_memory", 0) > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Optimizar uso de memoria")
        
        if stats.get("total_calls", 0) > 1000:
            recommendations.append("Considerar caché para resultados frecuentes")
        
        return "; ".join(recommendations) if recommendations else "Rendimiento aceptable"
    
    def generate_report(self) -> str:
        """Generar reporte de rendimiento"""
        report = f"# Reporte de Rendimiento\n\n"
        report += f"*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        stats = self.get_statistics()
        if not stats:
            return report + "No hay datos de rendimiento disponibles.\n"
        
        report += "## Resumen General\n\n"
        report += f"Funciones analizadas: {len(stats)}\n\n"
        
        report += "## Métricas por Función\n\n"
        report += "| Función | Llamadas | Tiempo Promedio (ms) | Memoria Promedio (MB) |\n"
        report += "|---------|----------|----------------------|----------------------|\n"
        
        for func_name, func_stats in sorted(stats.items(), key=lambda x: x[1].get("avg_time", 0), reverse=True):
            report += f"| {func_name} | {func_stats.get('total_calls', 0)} | "
            report += f"{func_stats.get('avg_time', 0) * 1000:.2f} | "
            report += f"{func_stats.get('avg_memory', 0) / 1024 / 1024:.2f} |\n"
        
        report += "\n"
        
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            report += "## Cuellos de Botella Detectados\n\n"
            for bottleneck in bottlenecks[:10]:  # Top 10
                report += f"### {bottleneck['function']}\n\n"
                report += f"- **Tiempo promedio**: {bottleneck['avg_time_ms']:.2f} ms\n"
                report += f"- **Llamadas**: {bottleneck['calls']}\n"
                report += f"- **Recomendación**: {bottleneck['recommendation']}\n\n"
        
        return report


# Instancia global
_performance_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Obtener instancia global del profiler"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


import asyncio

















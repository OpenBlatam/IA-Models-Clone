"""
Performance Profiler Module
===========================

Perfilado y análisis de rendimiento del agente.
Proporciona medición de tiempo, memoria, y análisis de cuellos de botella.
"""

import time
import functools
import asyncio
import tracemalloc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


class ProfilerMode(Enum):
    """Modo de perfilado."""
    TIME = "time"
    MEMORY = "memory"
    BOTH = "both"
    DISABLED = "disabled"


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento."""
    name: str
    duration: float
    memory_used: Optional[float] = None
    memory_peak: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "duration": self.duration,
            "memory_used": self.memory_used,
            "memory_peak": self.memory_peak,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class FunctionProfile:
    """Perfil de una función."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    total_memory: float = 0.0
    peak_memory: float = 0.0
    
    def update(self, duration: float, memory: Optional[float] = None) -> None:
        """Actualizar perfil con nueva medición."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.call_count
        
        if memory is not None:
            self.total_memory += memory
            self.peak_memory = max(self.peak_memory, memory)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "total_memory": self.total_memory,
            "peak_memory": self.peak_memory
        }


class PerformanceProfiler:
    """
    Perfilador de rendimiento.
    
    Proporciona funcionalidades para:
    - Medición de tiempo de ejecución
    - Medición de uso de memoria
    - Análisis de cuellos de botella
    - Perfiles de funciones
    - Reportes de rendimiento
    """
    
    def __init__(self, mode: ProfilerMode = ProfilerMode.BOTH, enabled: bool = True):
        """
        Inicializar perfilador.
        
        Args:
            mode: Modo de perfilado
            enabled: Si el perfilado está habilitado
        """
        self.mode = mode
        self.enabled = enabled
        self.metrics: List[PerformanceMetric] = []
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.active_profiles: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, tracemalloc.Snapshot] = {}
        self._memory_tracking = False
        
        if self.mode in [ProfilerMode.MEMORY, ProfilerMode.BOTH]:
            try:
                tracemalloc.start()
                self._memory_tracking = True
            except RuntimeError:
                logger.warning("Memory tracking already started or not available")
                self._memory_tracking = False
    
    @contextmanager
    def profile(self, name: str, context: Optional[Dict[str, Any]] = None):
        """
        Context manager para perfilado.
        
        Args:
            name: Nombre de la operación
            context: Contexto adicional
        """
        if not self.enabled or self.mode == ProfilerMode.DISABLED:
            yield
            return
        
        # Inicio de perfilado
        start_time = time.perf_counter()
        start_memory = None
        peak_memory = None
        
        if self._memory_tracking and self.mode in [ProfilerMode.MEMORY, ProfilerMode.BOTH]:
            start_memory = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            # Fin de perfilado
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if self._memory_tracking and self.mode in [ProfilerMode.MEMORY, ProfilerMode.BOTH]:
                end_memory = tracemalloc.take_snapshot()
                memory_used = sum(stat.size for stat in end_memory.compare_to(start_memory, 'lineno'))
                peak_memory = tracemalloc.get_traced_memory()[1]
            else:
                memory_used = None
                peak_memory = None
            
            # Crear métrica
            metric = PerformanceMetric(
                name=name,
                duration=duration,
                memory_used=memory_used / (1024 * 1024) if memory_used else None,  # MB
                memory_peak=peak_memory / (1024 * 1024) if peak_memory else None,  # MB
                context=context or {}
            )
            
            self.metrics.append(metric)
            
            # Actualizar perfil de función
            if name not in self.function_profiles:
                self.function_profiles[name] = FunctionProfile(function_name=name)
            
            self.function_profiles[name].update(
                duration,
                memory_used / (1024 * 1024) if memory_used else None
            )
    
    def profile_function(self, func: Callable):
        """
        Decorator para perfilado de función.
        
        Args:
            func: Función a perfilar
        """
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            with self.profile(func.__name__, {"args_count": len(args), "kwargs_count": len(kwargs)}):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.enabled:
                return await func(*args, **kwargs)
            
            with self.profile(func.__name__, {"args_count": len(args), "kwargs_count": len(kwargs)}):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def start_profile(self, name: str) -> None:
        """
        Iniciar perfilado manual.
        
        Args:
            name: Nombre del perfil
        """
        if not self.enabled:
            return
        
        self.active_profiles[name] = time.perf_counter()
        
        if self._memory_tracking and self.mode in [ProfilerMode.MEMORY, ProfilerMode.BOTH]:
            self.memory_snapshots[name] = tracemalloc.take_snapshot()
    
    def end_profile(self, name: str, context: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetric]:
        """
        Finalizar perfilado manual.
        
        Args:
            name: Nombre del perfil
            context: Contexto adicional
            
        Returns:
            Métrica de rendimiento o None
        """
        if not self.enabled or name not in self.active_profiles:
            return None
        
        start_time = self.active_profiles.pop(name)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        memory_used = None
        peak_memory = None
        
        if self._memory_tracking and self.mode in [ProfilerMode.MEMORY, ProfilerMode.BOTH]:
            if name in self.memory_snapshots:
                end_snapshot = tracemalloc.take_snapshot()
                start_snapshot = self.memory_snapshots.pop(name)
                memory_used = sum(stat.size for stat in end_snapshot.compare_to(start_snapshot, 'lineno'))
                peak_memory = tracemalloc.get_traced_memory()[1]
        
        metric = PerformanceMetric(
            name=name,
            duration=duration,
            memory_used=memory_used / (1024 * 1024) if memory_used else None,
            memory_peak=peak_memory / (1024 * 1024) if peak_memory else None,
            context=context or {}
        )
        
        self.metrics.append(metric)
        
        # Actualizar perfil
        if name not in self.function_profiles:
            self.function_profiles[name] = FunctionProfile(function_name=name)
        
        self.function_profiles[name].update(
            duration,
            memory_used / (1024 * 1024) if memory_used else None
        )
        
        return metric
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """
        Obtener métricas.
        
        Args:
            name: Filtrar por nombre
            limit: Límite de resultados
            
        Returns:
            Lista de métricas
        """
        filtered = self.metrics
        
        if name:
            filtered = [m for m in filtered if m.name == name]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_function_profile(self, function_name: str) -> Optional[FunctionProfile]:
        """
        Obtener perfil de una función.
        
        Args:
            function_name: Nombre de la función
            
        Returns:
            Perfil de función o None
        """
        return self.function_profiles.get(function_name)
    
    def get_all_profiles(self) -> Dict[str, FunctionProfile]:
        """
        Obtener todos los perfiles.
        
        Returns:
            Dict con todos los perfiles
        """
        return self.function_profiles.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de rendimiento.
        
        Returns:
            Dict con estadísticas
        """
        if not self.metrics:
            return {
                "total_measurements": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "functions_profiled": 0
            }
        
        total_time = sum(m.duration for m in self.metrics)
        avg_time = total_time / len(self.metrics)
        
        # Top funciones por tiempo
        function_times = defaultdict(float)
        for metric in self.metrics:
            function_times[metric.name] += metric.duration
        
        top_functions = sorted(
            function_times.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats = {
            "total_measurements": len(self.metrics),
            "total_time": total_time,
            "avg_time": avg_time,
            "min_time": min(m.duration for m in self.metrics),
            "max_time": max(m.duration for m in self.metrics),
            "functions_profiled": len(self.function_profiles),
            "top_functions_by_time": [
                {"name": name, "total_time": time}
                for name, time in top_functions
            ]
        }
        
        # Estadísticas de memoria si está disponible
        memory_metrics = [m for m in self.metrics if m.memory_used is not None]
        if memory_metrics:
            stats["memory"] = {
                "total_memory_used": sum(m.memory_used for m in memory_metrics),
                "avg_memory_used": sum(m.memory_used for m in memory_metrics) / len(memory_metrics),
                "peak_memory": max(m.memory_peak for m in memory_metrics if m.memory_peak)
            }
        
        return stats
    
    def get_bottlenecks(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """
        Identificar cuellos de botella.
        
        Args:
            threshold: Umbral de tiempo en segundos
            
        Returns:
            Lista de cuellos de botella
        """
        bottlenecks = []
        
        for profile in self.function_profiles.values():
            if profile.avg_time >= threshold or profile.max_time >= threshold:
                bottlenecks.append({
                    "function": profile.function_name,
                    "avg_time": profile.avg_time,
                    "max_time": profile.max_time,
                    "call_count": profile.call_count,
                    "total_time": profile.total_time
                })
        
        return sorted(bottlenecks, key=lambda x: x["avg_time"], reverse=True)
    
    def clear_metrics(self) -> None:
        """Limpiar todas las métricas."""
        self.metrics.clear()
        self.function_profiles.clear()
        self.active_profiles.clear()
        self.memory_snapshots.clear()
    
    def enable(self) -> None:
        """Habilitar perfilado."""
        self.enabled = True
        if self.mode in [ProfilerMode.MEMORY, ProfilerMode.BOTH] and not self._memory_tracking:
            try:
                tracemalloc.start()
                self._memory_tracking = True
            except RuntimeError:
                logger.warning("Memory tracking not available")
    
    def disable(self) -> None:
        """Deshabilitar perfilado."""
        self.enabled = False
    
    def set_mode(self, mode: ProfilerMode) -> None:
        """
        Establecer modo de perfilado.
        
        Args:
            mode: Nuevo modo
        """
        self.mode = mode
        
        if mode == ProfilerMode.DISABLED:
            self.disable()
        else:
            self.enable()
    
    def export_report(self, file_path: str, format: str = "json") -> bool:
        """
        Exportar reporte de rendimiento.
        
        Args:
            file_path: Ruta del archivo
            format: Formato (json, txt)
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            if format == "json":
                import json
                report = {
                    "statistics": self.get_statistics(),
                    "function_profiles": {
                        name: profile.to_dict()
                        for name, profile in self.function_profiles.items()
                    },
                    "bottlenecks": self.get_bottlenecks(),
                    "metrics": [m.to_dict() for m in self.metrics[-100:]]  # Últimas 100
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    stats = self.get_statistics()
                    f.write("Performance Report\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Total Measurements: {stats['total_measurements']}\n")
                    f.write(f"Total Time: {stats['total_time']:.2f}s\n")
                    f.write(f"Average Time: {stats['avg_time']:.4f}s\n")
                    f.write(f"Functions Profiled: {stats['functions_profiled']}\n\n")
                    
                    f.write("Top Functions by Time:\n")
                    for func in stats.get('top_functions_by_time', [])[:10]:
                        f.write(f"  {func['name']}: {func['total_time']:.4f}s\n")
                    
                    f.write("\nBottlenecks:\n")
                    for bottleneck in self.get_bottlenecks():
                        f.write(f"  {bottleneck['function']}: avg={bottleneck['avg_time']:.4f}s, "
                               f"max={bottleneck['max_time']:.4f}s, calls={bottleneck['call_count']}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}", exc_info=True)
            return False


def create_performance_profiler(
    mode: ProfilerMode = ProfilerMode.BOTH,
    enabled: bool = True
) -> PerformanceProfiler:
    """
    Factory function para crear PerformanceProfiler.
    
    Args:
        mode: Modo de perfilado
        enabled: Si está habilitado
        
    Returns:
        Instancia de PerformanceProfiler
    """
    return PerformanceProfiler(mode=mode, enabled=enabled)

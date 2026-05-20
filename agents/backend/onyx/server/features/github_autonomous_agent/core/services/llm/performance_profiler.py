"""
Performance Profiler - Profiling y análisis de performance.

Sigue principios de profiling en deep learning.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import time
import asyncio

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileResult:
    """Resultado de profiling."""
    function_name: str
    total_time: float
    call_count: int
    average_time: float
    min_time: float
    max_time: float
    total_tokens: int = 0
    total_cost: float = 0.0
    errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "function_name": self.function_name,
            "total_time": self.total_time,
            "call_count": self.call_count,
            "average_time": self.average_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "errors": self.errors,
            "tokens_per_second": (
                self.total_tokens / self.total_time
                if self.total_time > 0
                else 0.0
            )
        }


class PerformanceProfiler:
    """
    Profiler de performance para LLM Service.
    
    Sigue principios de profiling (como PyTorch profiler).
    """
    
    def __init__(self):
        """Inicializar profiler."""
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.enabled = True
        self.start_times: Dict[str, float] = {}
    
    def start(self, name: str) -> None:
        """
        Iniciar profiling de una función.
        
        Args:
            name: Nombre de la función/operación
        """
        if not self.enabled:
            return
        
        self.start_times[name] = time.perf_counter()
    
    def stop(
        self,
        name: str,
        tokens: int = 0,
        cost: float = 0.0,
        error: bool = False
    ) -> float:
        """
        Detener profiling y registrar métricas.
        
        Args:
            name: Nombre de la función/operación
            tokens: Tokens procesados (opcional)
            cost: Costo de la operación (opcional)
            error: Si hubo error (opcional)
            
        Returns:
            Duración en segundos
        """
        if not self.enabled or name not in self.start_times:
            return 0.0
        
        duration = time.perf_counter() - self.start_times.pop(name)
        self.profiles[name].append(duration)
        
        # Actualizar metadata
        if name not in self.metadata:
            self.metadata[name] = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "errors": 0
            }
        
        self.metadata[name]["total_tokens"] += tokens
        self.metadata[name]["total_cost"] += cost
        if error:
            self.metadata[name]["errors"] += 1
        
        return duration
    
    def profile_function(
        self,
        func: Callable,
        name: Optional[str] = None
    ) -> Callable:
        """
        Decorador para profiling de funciones.
        
        Args:
            func: Función a perfilar
            name: Nombre para el profiling (usa __name__ si None)
            
        Returns:
            Función decorada
        """
        func_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            self.start(func_name)
            try:
                result = func(*args, **kwargs)
                self.stop(func_name)
                return result
            except Exception as e:
                self.stop(func_name, error=True)
                raise
        
        return wrapper
    
    def profile_async_function(
        self,
        func: Callable,
        name: Optional[str] = None
    ) -> Callable:
        """
        Decorador para profiling de funciones async.
        
        Args:
            func: Función async a perfilar
            name: Nombre para el profiling
            
        Returns:
            Función async decorada
        """
        func_name = name or func.__name__
        
        async def wrapper(*args, **kwargs):
            self.start(func_name)
            try:
                result = await func(*args, **kwargs)
                self.stop(func_name)
                return result
            except Exception as e:
                self.stop(func_name, error=True)
                raise
        
        return wrapper
    
    def get_profile(
        self,
        name: str
    ) -> Optional[ProfileResult]:
        """
        Obtener perfil de una función.
        
        Args:
            name: Nombre de la función
            
        Returns:
            ProfileResult o None
        """
        if name not in self.profiles or not self.profiles[name]:
            return None
        
        times = self.profiles[name]
        meta = self.metadata.get(name, {})
        
        return ProfileResult(
            function_name=name,
            total_time=sum(times),
            call_count=len(times),
            average_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
            total_tokens=meta.get("total_tokens", 0),
            total_cost=meta.get("total_cost", 0.0),
            errors=meta.get("errors", 0)
        )
    
    def get_all_profiles(self) -> Dict[str, ProfileResult]:
        """
        Obtener todos los perfiles.
        
        Returns:
            Diccionario con todos los perfiles
        """
        return {
            name: self.get_profile(name)
            for name in self.profiles.keys()
            if self.profiles[name]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de performance.
        
        Returns:
            Diccionario con resumen
        """
        all_profiles = self.get_all_profiles()
        
        if not all_profiles:
            return {
                "total_functions": 0,
                "total_time": 0.0,
                "total_calls": 0
            }
        
        total_time = sum(p.total_time for p in all_profiles.values())
        total_calls = sum(p.call_count for p in all_profiles.values())
        total_tokens = sum(p.total_tokens for p in all_profiles.values())
        total_cost = sum(p.total_cost for p in all_profiles.values())
        
        # Funciones más lentas
        slowest = sorted(
            all_profiles.values(),
            key=lambda x: x.average_time,
            reverse=True
        )[:5]
        
        # Funciones más llamadas
        most_called = sorted(
            all_profiles.values(),
            key=lambda x: x.call_count,
            reverse=True
        )[:5]
        
        return {
            "total_functions": len(all_profiles),
            "total_time": total_time,
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_time_per_call": total_time / total_calls if total_calls > 0 else 0.0,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
            "slowest_functions": [p.to_dict() for p in slowest],
            "most_called_functions": [p.to_dict() for p in most_called]
        }
    
    def reset(self) -> None:
        """Resetear todos los perfiles."""
        self.profiles.clear()
        self.metadata.clear()
        self.start_times.clear()
        logger.info("Profiles reseteados")
    
    def export_report(self, filepath: str) -> None:
        """
        Exportar reporte de performance.
        
        Args:
            filepath: Ruta donde guardar el reporte
        """
        summary = self.get_summary()
        all_profiles = {
            name: profile.to_dict()
            for name, profile in self.get_all_profiles().items()
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "profiles": all_profiles
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reporte de performance exportado a {filepath}")


# Instancia global
_performance_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Obtener instancia global del profiler."""
    return _performance_profiler




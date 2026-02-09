"""
Performance Analysis - Análisis de Rendimiento
===============================================

Utilidades para analizar y optimizar el rendimiento del sistema.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Perfil de rendimiento de una operación"""
    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    p50_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    errors: int = 0
    last_call: Optional[datetime] = None
    call_times: List[float] = field(default_factory=list)
    
    def add_call(self, duration: float, error: bool = False):
        """Agregar llamada al perfil"""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.call_times.append(duration)
        self.last_call = datetime.now()
        
        if error:
            self.errors += 1
        
        # Calcular percentiles
        if self.call_times:
            sorted_times = sorted(self.call_times)
            n = len(sorted_times)
            self.avg_time = self.total_time / self.total_calls
            self.p50_time = sorted_times[n // 2] if n > 0 else 0
            self.p95_time = sorted_times[int(n * 0.95)] if n > 1 else sorted_times[0]
            self.p99_time = sorted_times[int(n * 0.99)] if n > 1 else sorted_times[-1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "p50_time": self.p50_time,
            "p95_time": self.p95_time,
            "p99_time": self.p99_time,
            "errors": self.errors,
            "error_rate": self.errors / self.total_calls if self.total_calls > 0 else 0,
            "last_call": self.last_call.isoformat() if self.last_call else None
        }


class PerformanceAnalyzer:
    """
    Analizador de rendimiento.
    
    Recopila y analiza métricas de rendimiento del sistema.
    """
    
    def __init__(self, max_samples: int = 10000):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.max_samples = max_samples
    
    def profile_function(
        self,
        func_name: Optional[str] = None,
        track_errors: bool = True
    ):
        """
        Decorador para perfilar función.
        
        Args:
            func_name: Nombre personalizado (opcional)
            track_errors: Si trackear errores
        """
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start = time.time()
                    error = False
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception:
                        error = True
                        raise
                    finally:
                        duration = time.time() - start
                        self.record_call(name, duration, error=error and track_errors)
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start = time.time()
                    error = False
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception:
                        error = True
                        raise
                    finally:
                        duration = time.time() - start
                        self.record_call(name, duration, error=error and track_errors)
                
                return sync_wrapper
        
        return decorator
    
    def record_call(
        self,
        name: str,
        duration: float,
        error: bool = False
    ) -> None:
        """
        Registrar llamada.
        
        Args:
            name: Nombre de la operación
            duration: Duración en segundos
            error: Si hubo error
        """
        if name not in self.profiles:
            self.profiles[name] = PerformanceProfile(name=name)
        
        profile = self.profiles[name]
        profile.add_call(duration, error)
        
        # Limitar muestras
        if len(profile.call_times) > self.max_samples:
            profile.call_times = profile.call_times[-self.max_samples:]
    
    def get_profile(self, name: str) -> Optional[PerformanceProfile]:
        """Obtener perfil de rendimiento"""
        return self.profiles.get(name)
    
    def get_all_profiles(self) -> Dict[str, PerformanceProfile]:
        """Obtener todos los perfiles"""
        return self.profiles.copy()
    
    def get_slowest_operations(self, limit: int = 10) -> List[PerformanceProfile]:
        """
        Obtener operaciones más lentas.
        
        Args:
            limit: Número de operaciones a retornar
            
        Returns:
            Lista de perfiles ordenados por tiempo promedio
        """
        profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.avg_time,
            reverse=True
        )
        return profiles[:limit]
    
    def get_most_called_operations(self, limit: int = 10) -> List[PerformanceProfile]:
        """
        Obtener operaciones más llamadas.
        
        Args:
            limit: Número de operaciones a retornar
            
        Returns:
            Lista de perfiles ordenados por número de llamadas
        """
        profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.total_calls,
            reverse=True
        )
        return profiles[:limit]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de rendimiento.
        
        Returns:
            Diccionario con resumen
        """
        total_calls = sum(p.total_calls for p in self.profiles.values())
        total_time = sum(p.total_time for p in self.profiles.values())
        total_errors = sum(p.errors for p in self.profiles.values())
        
        return {
            "total_operations": len(self.profiles),
            "total_calls": total_calls,
            "total_time": total_time,
            "avg_time_per_call": total_time / total_calls if total_calls > 0 else 0,
            "total_errors": total_errors,
            "error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "slowest_operations": [
                p.to_dict() for p in self.get_slowest_operations(5)
            ],
            "most_called_operations": [
                p.to_dict() for p in self.get_most_called_operations(5)
            ]
        }
    
    def clear(self) -> None:
        """Limpiar todos los perfiles"""
        self.profiles.clear()
        logger.info("🧹 Performance profiles cleared")


# Importar asyncio y wraps
import asyncio
from functools import wraps





"""
Servicio de profiling de rendimiento para análisis detallado.
"""

import time
import functools
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileEntry:
    """Entrada de perfil."""
    name: str
    duration: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['ProfileEntry'] = field(default_factory=list)


class PerformanceProfiler:
    """
    Profiler de rendimiento con mejoras.
    
    Attributes:
        max_entries: Número máximo de entradas
        entries: Lista de entradas de perfil
        active_profiles: Perfiles activos por ID
        stats: Estadísticas por nombre de operación
        enabled: Si el profiler está habilitado
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Inicializar profiler con validaciones.
        
        Args:
            max_entries: Número máximo de entradas a mantener (debe ser entero positivo)
            
        Raises:
            ValueError: Si max_entries es inválido
        """
        # Validación
        if not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError(f"max_entries debe ser un entero positivo, recibido: {max_entries}")
        
        self.max_entries = max_entries
        self.entries: List[ProfileEntry] = []
        self.active_profiles: Dict[str, ProfileEntry] = {}
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "avg_time": 0.0
        })
        self.enabled = True
        
        logger.info(f"✅ PerformanceProfiler inicializado: max_entries={max_entries}")
    
    def start(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Iniciar profiling de una operación con validaciones.
        
        Args:
            name: Nombre de la operación (debe ser string no vacío)
            metadata: Metadata adicional (opcional, debe ser diccionario si se proporciona)
            
        Returns:
            ID del perfil iniciado (string vacío si está deshabilitado)
            
        Raises:
            ValueError: Si name es inválido
        """
        # Validación
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata debe ser un diccionario si se proporciona, recibido: {type(metadata)}")
        
        if not self.enabled:
            logger.debug(f"Profiler deshabilitado, ignorando start para '{name}'")
            return ""
        
        name = name.strip()
        profile_id = f"{name}_{time.time()}_{len(self.active_profiles)}"
        entry = ProfileEntry(
            name=name,
            duration=0.0,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.active_profiles[profile_id] = entry
        logger.debug(f"⏱️  Profiling iniciado: {name} (profile_id: {profile_id[:20]}...)")
        return profile_id
    
    def stop(self, profile_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Detener profiling de una operación.
        
        Args:
            profile_id: ID del perfil
            metadata: Metadata adicional
            
        Returns:
            Duración en segundos o None si no se encontró
        """
        if not profile_id or profile_id not in self.active_profiles:
            return None
        
        entry = self.active_profiles.pop(profile_id)
        duration = (datetime.now() - entry.timestamp).total_seconds()
        entry.duration = duration
        
        if metadata:
            entry.metadata.update(metadata)
        
        # Actualizar estadísticas
        stats = self.stats[entry.name]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        stats["avg_time"] = stats["total_time"] / stats["count"]
        
        # Agregar a entradas
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        
        return duration
    
    def profile_function(self, name: Optional[str] = None):
        """
        Decorador para profiling de funciones.
        
        Args:
            name: Nombre personalizado (opcional)
            
        Returns:
            Decorador
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                profile_id = self.start(func_name)
                try:
                    result = await func(*args, **kwargs)
                    self.stop(profile_id, {"success": True})
                    return result
                except Exception as e:
                    self.stop(profile_id, {"success": False, "error": str(e)})
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                profile_id = self.start(func_name)
                try:
                    result = func(*args, **kwargs)
                    self.stop(profile_id, {"success": True})
                    return result
                except Exception as e:
                    self.stop(profile_id, {"success": False, "error": str(e)})
                    raise
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de profiling.
        
        Args:
            name: Nombre de la operación (opcional, todas si None)
            
        Returns:
            Estadísticas
        """
        if name:
            return self.stats.get(name, {})
        return dict(self.stats)
    
    def get_recent_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener entradas recientes con validaciones.
        
        Args:
            limit: Número máximo de entradas (debe ser entero positivo)
            
        Returns:
            Lista de entradas recientes
            
        Raises:
            ValueError: Si limit es inválido
        """
        # Validación
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit debe ser un entero positivo, recibido: {limit}")
        
        # Limitar a máximo razonable
        if limit > 10000:
            logger.warning(f"limit muy alto ({limit}), limitando a 10000")
            limit = 10000
        
        recent = self.entries[-limit:]
        result = [
            {
                "name": e.name,
                "duration": e.duration,
                "timestamp": e.timestamp.isoformat(),
                "metadata": e.metadata
            }
            for e in recent
        ]
        
        logger.debug(f"Obtenidas {len(result)} entradas recientes (limit: {limit})")
        return result
    
    def get_slowest_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener operaciones más lentas.
        
        Args:
            limit: Número máximo de operaciones
            
        Returns:
            Lista de operaciones ordenadas por duración
        """
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.duration,
            reverse=True
        )
        return [
            {
                "name": e.name,
                "duration": e.duration,
                "timestamp": e.timestamp.isoformat(),
                "metadata": e.metadata
            }
            for e in sorted_entries[:limit]
        ]
    
    def reset(self) -> None:
        """Resetear profiler."""
        self.entries.clear()
        self.active_profiles.clear()
        self.stats.clear()
        logger.info("Performance profiler reseteado")
    
    def enable(self) -> None:
        """Habilitar profiling."""
        self.enabled = True
    
    def disable(self) -> None:
        """Deshabilitar profiling."""
        self.enabled = False


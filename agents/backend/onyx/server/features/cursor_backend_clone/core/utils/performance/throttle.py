"""
Throttle - Sistema de throttling para operaciones costosas
==========================================================

Sistema para limitar la frecuencia de operaciones costosas,
previniendo sobrecarga del sistema.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ThrottleStrategy(Enum):
    """Estrategias de throttling"""
    FIXED = "fixed"  # Delay fijo entre operaciones
    ADAPTIVE = "adaptive"  # Delay adaptativo basado en carga
    EXPONENTIAL = "exponential"  # Delay exponencial creciente


@dataclass
class ThrottleStats:
    """Estadísticas de throttling"""
    total_operations: int = 0
    throttled_operations: int = 0
    total_wait_time: float = 0.0
    avg_wait_time: float = 0.0
    last_operation_time: Optional[datetime] = None


class Throttler:
    """
    Throttler para limitar frecuencia de operaciones.
    
    Útil para operaciones costosas que no deben ejecutarse
    con demasiada frecuencia (ej: llamadas a APIs externas,
    operaciones de I/O pesadas, etc.)
    """
    
    def __init__(
        self,
        min_interval: float = 1.0,
        strategy: ThrottleStrategy = ThrottleStrategy.FIXED,
        max_wait: float = 60.0
    ):
        """
        Inicializar throttler.
        
        Args:
            min_interval: Intervalo mínimo entre operaciones en segundos
            strategy: Estrategia de throttling
            max_wait: Tiempo máximo de espera antes de fallar
        """
        self.min_interval = min_interval
        self.strategy = strategy
        self.max_wait = max_wait
        self.last_operation_time: Optional[float] = None
        self.operation_count = 0
        self.stats = ThrottleStats()
        self._lock = asyncio.Lock()
        self._adaptive_delay = min_interval
    
    async def throttle(self) -> float:
        """
        Aplicar throttling y retornar tiempo de espera.
        
        Returns:
            Tiempo de espera en segundos
        """
        async with self._lock:
            now = time.time()
            wait_time = 0.0
            
            if self.last_operation_time is not None:
                elapsed = now - self.last_operation_time
                
                if self.strategy == ThrottleStrategy.FIXED:
                    wait_time = max(0, self.min_interval - elapsed)
                elif self.strategy == ThrottleStrategy.ADAPTIVE:
                    wait_time = max(0, self._adaptive_delay - elapsed)
                elif self.strategy == ThrottleStrategy.EXPONENTIAL:
                    # Aumentar delay con cada operación
                    wait_time = max(0, self.min_interval * (1.5 ** self.operation_count) - elapsed)
                
                if wait_time > 0:
                    if wait_time > self.max_wait:
                        raise asyncio.TimeoutError(
                            f"Throttle wait time ({wait_time:.2f}s) exceeds max wait ({self.max_wait}s)"
                        )
                    
                    self.stats.throttled_operations += 1
                    self.stats.total_wait_time += wait_time
                    await asyncio.sleep(wait_time)
                    now = time.time()
            
            self.last_operation_time = now
            self.operation_count += 1
            self.stats.total_operations += 1
            self.stats.last_operation_time = datetime.now()
            
            # Actualizar delay adaptativo
            if self.strategy == ThrottleStrategy.ADAPTIVE:
                if wait_time > 0:
                    # Aumentar delay si hubo throttling
                    self._adaptive_delay = min(self._adaptive_delay * 1.1, self.min_interval * 5)
                else:
                    # Reducir delay si no hubo throttling
                    self._adaptive_delay = max(self._adaptive_delay * 0.9, self.min_interval)
            
            return wait_time
    
    def get_stats(self) -> ThrottleStats:
        """Obtener estadísticas de throttling"""
        if self.stats.total_operations > 0:
            self.stats.avg_wait_time = (
                self.stats.total_wait_time / self.stats.throttled_operations
                if self.stats.throttled_operations > 0 else 0.0
            )
        return self.stats
    
    def reset(self) -> None:
        """Resetear throttler"""
        self.last_operation_time = None
        self.operation_count = 0
        self.stats = ThrottleStats()
        self._adaptive_delay = self.min_interval


def throttle_async(
    min_interval: float = 1.0,
    strategy: ThrottleStrategy = ThrottleStrategy.FIXED,
    max_wait: float = 60.0
):
    """
    Decorador para aplicar throttling a funciones async.
    
    Args:
        min_interval: Intervalo mínimo entre llamadas en segundos
        strategy: Estrategia de throttling
        max_wait: Tiempo máximo de espera
        
    Example:
        @throttle_async(min_interval=2.0, strategy=ThrottleStrategy.ADAPTIVE)
        async def expensive_operation():
            ...
    """
    throttler = Throttler(min_interval, strategy, max_wait)
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await throttler.throttle()
            return await func(*args, **kwargs)
        
        # Agregar método para obtener stats
        wrapper.get_throttle_stats = lambda: throttler.get_stats()
        wrapper.reset_throttle = lambda: throttler.reset()
        
        return wrapper
    return decorator


class OperationThrottler:
    """
    Throttler por tipo de operación.
    
    Permite diferentes límites de throttling para diferentes
    tipos de operaciones.
    """
    
    def __init__(self):
        self.throttlers: Dict[str, Throttler] = {}
        self._lock = asyncio.Lock()
    
    async def throttle(
        self,
        operation_type: str,
        min_interval: float = 1.0,
        strategy: ThrottleStrategy = ThrottleStrategy.FIXED
    ) -> float:
        """
        Aplicar throttling para un tipo de operación.
        
        Args:
            operation_type: Tipo de operación
            min_interval: Intervalo mínimo (si no existe throttler para este tipo)
            strategy: Estrategia (si no existe throttler para este tipo)
            
        Returns:
            Tiempo de espera en segundos
        """
        async with self._lock:
            if operation_type not in self.throttlers:
                self.throttlers[operation_type] = Throttler(min_interval, strategy)
            
            return await self.throttlers[operation_type].throttle()
    
    def get_stats(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de throttling.
        
        Args:
            operation_type: Tipo de operación (None para todas)
            
        Returns:
            Estadísticas de throttling
        """
        if operation_type:
            if operation_type in self.throttlers:
                stats = self.throttlers[operation_type].get_stats()
                return {
                    operation_type: {
                        "total_operations": stats.total_operations,
                        "throttled_operations": stats.throttled_operations,
                        "avg_wait_time": stats.avg_wait_time,
                        "last_operation_time": stats.last_operation_time.isoformat() if stats.last_operation_time else None
                    }
                }
            return {}
        else:
            return {
                op_type: {
                    "total_operations": throttler.stats.total_operations,
                    "throttled_operations": throttler.stats.throttled_operations,
                    "avg_wait_time": throttler.stats.avg_wait_time,
                    "last_operation_time": throttler.stats.last_operation_time.isoformat() if throttler.stats.last_operation_time else None
                }
                for op_type, throttler in self.throttlers.items()
            }
    
    def reset(self, operation_type: Optional[str] = None) -> None:
        """Resetear throttler(s)"""
        if operation_type:
            if operation_type in self.throttlers:
                self.throttlers[operation_type].reset()
        else:
            for throttler in self.throttlers.values():
                throttler.reset()





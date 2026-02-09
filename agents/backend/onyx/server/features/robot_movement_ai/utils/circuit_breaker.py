"""
Circuit Breaker - Circuit breaker pattern
=========================================

Implementación del patrón circuit breaker para servicios externos.
"""

import asyncio
import time
from typing import Callable, TypeVar, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..core.exceptions import BaseRobotException

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_timeout: float = 30.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker para proteger servicios externos."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Inicializar circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            config: Configuración
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._stats: Dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_opens": 0,
            "circuit_closes": 0
        }
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Ejecutar función a través del circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado de la función
        
        Raises:
            Exception: Si el circuit está abierto o la función falla
        """
        self._stats["total_calls"] += 1
        
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit {self.name} moved to HALF_OPEN")
            else:
                raise BaseRobotException(
                    f"Circuit {self.name} is OPEN",
                    error_code="CIRCUIT_OPEN",
                    details={"circuit_name": self.name, "state": "open"}
                )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            
            self._handle_success()
            return result
        
        except Exception as e:
            self._handle_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Verificar si se debe intentar resetear el circuit."""
        if self._opened_at is None:
            return False
        
        elapsed = (datetime.now() - self._opened_at).total_seconds()
        return elapsed >= self.config.half_open_timeout
    
    def _handle_success(self):
        """Manejar llamada exitosa."""
        self._stats["successful_calls"] += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._opened_at = None
                self._stats["circuit_closes"] += 1
                logger.info(f"Circuit {self.name} moved to CLOSED")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def _handle_failure(self, exception: Exception):
        """Manejar llamada fallida."""
        self._stats["failed_calls"] += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = datetime.now()
            logger.warning(f"Circuit {self.name} moved to OPEN from HALF_OPEN")
        
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = datetime.now()
                self._stats["circuit_opens"] += 1
                logger.warning(
                    f"Circuit {self.name} moved to OPEN "
                    f"(failures: {self._failure_count})"
                )
    
    def get_state(self) -> CircuitState:
        """Obtener estado actual."""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "stats": self._stats.copy()
        }
    
    def reset(self):
        """Resetear circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._opened_at = None
        logger.info(f"Circuit {self.name} manually reset")


class CircuitBreakerManager:
    """Gestor de circuit breakers."""
    
    def __init__(self):
        """Inicializar gestor."""
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Obtener o crear circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            config: Configuración
        
        Returns:
            Circuit breaker
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas de todos los circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }


_global_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Obtener gestor global de circuit breakers."""
    return _global_manager


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorador para usar circuit breaker.
    
    Args:
        name: Nombre del circuit breaker
        config: Configuración
    
    Example:
        @circuit_breaker("robot_connection")
        async def connect_to_robot():
            ...
    """
    breaker = get_circuit_breaker_manager().get_breaker(name, config)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        from functools import wraps
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await breaker.call(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                async def async_func(*args, **kwargs):
                    return func(*args, **kwargs)
                return asyncio.run(breaker.call(async_func, *args, **kwargs))
            return sync_wrapper
    
    return decorator


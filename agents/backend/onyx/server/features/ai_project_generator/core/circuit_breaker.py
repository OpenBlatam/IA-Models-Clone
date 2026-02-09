"""
Circuit Breaker - Implementación de circuit breaker pattern
===========================================================

Implementa el patrón circuit breaker para comunicación resiliente
entre servicios siguiendo mejores prácticas de microservicios.
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from threading import Lock
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Estadísticas del circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit Breaker para proteger contra fallos en cascada.
    
    Implementa el patrón circuit breaker:
    - CLOSED: Funciona normalmente
    - OPEN: Rechaza requests después de umbral de fallos
    - HALF_OPEN: Prueba si el servicio se recuperó
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "default"
    ):
        """
        Args:
            failure_threshold: Número de fallos antes de abrir
            timeout: Tiempo en segundos antes de intentar half-open
            expected_exception: Excepción que cuenta como fallo
            name: Nombre del circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.stats = CircuitBreakerStats()
        self._lock = Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función con protección de circuit breaker"""
        with self._lock:
            # Verificar si debemos intentar half-open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    self.stats.rejected_requests += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Rejecting request. Will retry after {self.timeout}s"
                    )
        
        # Intentar ejecutar
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función async con protección de circuit breaker"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    self.stats.rejected_requests += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Rejecting request. Will retry after {self.timeout}s"
                    )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Verifica si debemos intentar reset (half-open)"""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.timeout
    
    def _on_success(self):
        """Maneja éxito"""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Éxito en half-open, cerrar circuit breaker
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.stats.state_changes += 1
                logger.info(f"Circuit breaker {self.name} CLOSED after successful recovery")
            elif self.state == CircuitState.CLOSED:
                # Resetear contador de fallos en caso de éxito
                self.failure_count = 0
    
    def _on_failure(self):
        """Maneja fallo"""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.stats.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Fallo en half-open, volver a abrir
                self.state = CircuitState.OPEN
                self.stats.state_changes += 1
                logger.warning(f"Circuit breaker {self.name} OPENED after failure in HALF_OPEN")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    # Abrir circuit breaker
                    self.state = CircuitState.OPEN
                    self.stats.state_changes += 1
                    logger.warning(
                        f"Circuit breaker {self.name} OPENED after {self.failure_count} failures"
                    )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del circuit breaker"""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "timeout": self.timeout,
                "stats": {
                    "total_requests": self.stats.total_requests,
                    "successful_requests": self.stats.successful_requests,
                    "failed_requests": self.stats.failed_requests,
                    "rejected_requests": self.stats.rejected_requests,
                    "state_changes": self.stats.state_changes,
                    "last_failure_time": self.stats.last_failure_time,
                    "last_success_time": self.stats.last_success_time,
                }
            }
    
    def reset(self):
        """Resetea el circuit breaker manualmente"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerOpenError(Exception):
    """Excepción cuando el circuit breaker está abierto"""
    pass


# Circuit breakers globales
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = Lock()


def get_circuit_breaker(name: str = "default", **kwargs) -> CircuitBreaker:
    """Obtiene o crea un circuit breaker"""
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return _circuit_breakers[name]


def circuit_breaker(name: str = "default", **kwargs):
    """Decorator para aplicar circuit breaker a una función"""
    breaker = get_circuit_breaker(name, **kwargs)
    
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kw):
                return await breaker.call_async(func, *args, **kw)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kw):
                return breaker.call(func, *args, **kw)
            return sync_wrapper
    
    return decorator


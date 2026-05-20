"""
Circuit Breaker - Patrón Circuit Breaker para resiliencia
==========================================================

Implementa el patrón Circuit Breaker para manejar fallos
en servicios externos y prevenir cascading failures.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del Circuit Breaker"""
    CLOSED = "closed"  # Normal, permitir requests
    OPEN = "open"  # Fallando, rechazar requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


@dataclass
class CircuitBreakerConfig:
    """Configuración del Circuit Breaker"""
    failure_threshold: int = 5  # Número de fallos antes de abrir
    success_threshold: int = 2  # Número de éxitos para cerrar desde half-open
    timeout: float = 60.0  # Tiempo en segundos antes de intentar half-open
    expected_exception: type = Exception  # Excepción esperada que cuenta como fallo


class CircuitBreaker:
    """
    Circuit Breaker para proteger llamadas a servicios externos.
    
    Implementa el patrón Circuit Breaker:
    - CLOSED: Funciona normalmente
    - OPEN: Demasiados fallos, rechazar requests
    - HALF_OPEN: Probando si el servicio se recuperó
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre del circuit breaker (para logging).
            config: Configuración (opcional).
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
        
        logger.info(f"CircuitBreaker '{name}' initialized")
    
    def _reset(self) -> None:
        """Resetear contadores."""
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def _should_attempt_reset(self) -> bool:
        """Verificar si debería intentar reset (half-open)."""
        if self.state != CircuitState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.timeout
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar función con protección de Circuit Breaker.
        
        Args:
            func: Función a ejecutar (puede ser async o sync).
            *args: Argumentos posicionales.
            **kwargs: Argumentos con nombre.
        
        Returns:
            Resultado de la función.
        
        Raises:
            CircuitBreakerOpenError: Si el circuit está abierto.
            Exception: Si la función falla.
        """
        # Verificar si debería intentar reset
        if self._should_attempt_reset():
            self.state = CircuitState.HALF_OPEN
            self._reset()
            self.last_state_change = time.time()
            logger.info(f"CircuitBreaker '{self.name}' moved to HALF_OPEN")
        
        # Si está abierto, rechazar
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"CircuitBreaker '{self.name}' is OPEN. "
                f"Last failure: {self.last_failure_time}"
            )
        
        # Ejecutar función
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Éxito
            self._on_success()
            return result
        
        except self.config.expected_exception as e:
            # Fallo esperado
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Manejar éxito."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self._reset()
                self.last_state_change = time.time()
                logger.info(f"CircuitBreaker '{self.name}' moved to CLOSED")
        else:
            # En CLOSED, resetear contador de fallos
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Manejar fallo."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Fallo en half-open, volver a abrir
            self.state = CircuitState.OPEN
            self._reset()
            self.last_state_change = time.time()
            logger.warning(f"CircuitBreaker '{self.name}' moved back to OPEN")
        
        elif self.failure_count >= self.config.failure_threshold:
            # Demasiados fallos, abrir circuit
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            logger.error(
                f"CircuitBreaker '{self.name}' moved to OPEN "
                f"after {self.failure_count} failures"
            )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del Circuit Breaker.
        
        Returns:
            Diccionario con estado actual.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
            "time_since_last_change": time.time() - self.last_state_change
        }


class CircuitBreakerOpenError(Exception):
    """
    Excepción cuando el Circuit Breaker está abierto.
    
    Se lanza cuando se intenta ejecutar una operación mientras el circuit breaker
    está en estado abierto (después de múltiples fallos).
    """
    
    def __init__(self, service_name: str = "unknown", failure_count: int = 0, 
                 last_failure_time: float = None, retry_after: float = None):
        """
        Initialize circuit breaker open error.
        
        Args:
            service_name: Name of the service with open circuit breaker
            failure_count: Number of failures that caused the circuit to open
            last_failure_time: Timestamp of last failure
            retry_after: Seconds until circuit breaker will retry
        """
        message = f"Circuit breaker is OPEN for service '{service_name}'"
        if failure_count > 0:
            message += f" (after {failure_count} failures)"
        if retry_after:
            message += f". Retry after {retry_after:.1f} seconds"
        
        super().__init__(message)
        self.service_name = service_name
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time
        self.retry_after = retry_after


# Circuit breakers globales para servicios comunes
_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Obtener o crear Circuit Breaker por nombre.
    
    Args:
        name: Nombre del circuit breaker.
        config: Configuración (opcional).
    
    Returns:
        Circuit Breaker.
    """
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, config)
    return _breakers[name]





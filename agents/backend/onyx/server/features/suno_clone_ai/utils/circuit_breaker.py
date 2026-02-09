"""
Circuit Breaker pattern para resiliencia en llamadas externas
"""

import logging
import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""
    failure_threshold: int = 5  # Número de fallos antes de abrir
    success_threshold: int = 2  # Número de éxitos para cerrar desde half-open
    timeout: float = 60.0  # Tiempo en segundos antes de intentar half-open
    expected_exception: type = Exception  # Excepción esperada que cuenta como fallo


@dataclass
class CircuitBreakerStats:
    """Estadísticas del circuit breaker"""
    failures: int = 0
    successes: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """Implementación del patrón Circuit Breaker"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta una función con protección del circuit breaker"""
        self.stats.total_calls += 1
        
        # Verificar estado
        if self.stats.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.stats.state = CircuitState.HALF_OPEN
                self.stats.successes = 0
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.stats.last_failure_time}"
                )
        
        # Intentar ejecutar
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Determina si se debe intentar resetear el circuit breaker"""
        if not self.stats.last_failure_time:
            return False
        
        elapsed = time.time() - self.stats.last_failure_time
        return elapsed >= self.config.timeout
    
    def _on_success(self):
        """Maneja un éxito"""
        self.stats.total_successes += 1
        
        if self.stats.state == CircuitState.HALF_OPEN:
            self.stats.successes += 1
            if self.stats.successes >= self.config.success_threshold:
                self.stats.state = CircuitState.CLOSED
                self.stats.failures = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED after recovery")
        elif self.stats.state == CircuitState.CLOSED:
            self.stats.failures = 0  # Reset failure count on success
    
    def _on_failure(self):
        """Maneja un fallo"""
        self.stats.total_failures += 1
        self.stats.failures += 1
        self.stats.last_failure_time = time.time()
        
        if self.stats.state == CircuitState.HALF_OPEN:
            self.stats.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' OPENED (failed in half-open)")
        elif self.stats.state == CircuitState.CLOSED:
            if self.stats.failures >= self.config.failure_threshold:
                self.stats.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED "
                    f"(failures: {self.stats.failures})"
                )
    
    def reset(self):
        """Resetea manualmente el circuit breaker"""
        self.stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del circuit breaker"""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "total_calls": self.stats.total_calls,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "last_failure_time": self.stats.last_failure_time,
            "failure_rate": (
                self.stats.total_failures / self.stats.total_calls
                if self.stats.total_calls > 0 else 0
            )
        }


class CircuitBreakerOpenError(Exception):
    """Excepción lanzada cuando el circuit breaker está abierto"""
    pass


# Circuit breakers globales
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Obtiene o crea un circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator para aplicar circuit breaker a una función"""
    cb = get_circuit_breaker(name, config)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


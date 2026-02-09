"""
Circuit Breaker Pattern para servicios resilientes
Implementa circuit breaker con estados: CLOSED, OPEN, HALF_OPEN
"""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from functools import wraps
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"  # Funcionando normalmente
    OPEN = "open"  # Fallando, rechazando requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""
    failure_threshold: int = 5  # Número de fallos antes de abrir
    success_threshold: int = 2  # Número de éxitos para cerrar desde half-open
    timeout: float = 60.0  # Tiempo en segundos antes de intentar half-open
    expected_exception: type = Exception  # Excepción esperada que cuenta como fallo


@dataclass
class CircuitBreakerState:
    """Estado interno del circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit Breaker para proteger servicios externos
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state_obj = CircuitBreakerState()
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self._lock = False  # Simple lock para thread safety básico
    
    def _should_attempt_request(self) -> bool:
        """Determina si se debe intentar la request basado en el estado"""
        current_time = time.time()
        
        if self.state_obj.state == CircuitState.CLOSED:
            return True
        
        elif self.state_obj.state == CircuitState.OPEN:
            # Verificar si ha pasado suficiente tiempo para intentar half-open
            if self.state_obj.last_failure_time:
                elapsed = current_time - self.state_obj.last_failure_time
                if elapsed >= self.config.timeout:
                    self.state_obj.state = CircuitState.HALF_OPEN
                    self.state_obj.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
            return False
        
        elif self.state_obj.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self):
        """Registra un éxito"""
        self.state_obj.last_success_time = time.time()
        self.state_obj.success_count += 1
        self.state_obj.total_successes += 1
        
        if self.state_obj.state == CircuitState.HALF_OPEN:
            if self.state_obj.success_count >= self.config.success_threshold:
                self.state_obj.state = CircuitState.CLOSED
                self.state_obj.failure_count = 0
                self.state_obj.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED (service recovered)")
                if self.on_close_callback:
                    self.on_close_callback()
        elif self.state_obj.state == CircuitState.CLOSED:
            # Reset failure count en CLOSED si hay éxito
            self.state_obj.failure_count = 0
    
    def _record_failure(self):
        """Registra un fallo"""
        self.state_obj.last_failure_time = time.time()
        self.state_obj.failure_count += 1
        self.state_obj.total_failures += 1
        
        if self.state_obj.state == CircuitState.HALF_OPEN:
            # Cualquier fallo en half-open vuelve a abrir
            self.state_obj.state = CircuitState.OPEN
            self.state_obj.success_count = 0
            logger.warning(f"Circuit breaker '{self.name}' OPEN (service still failing)")
            if self.on_open_callback:
                self.on_open_callback()
        
        elif self.state_obj.state == CircuitState.CLOSED:
            if self.state_obj.failure_count >= self.config.failure_threshold:
                self.state_obj.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN "
                    f"(failure threshold {self.config.failure_threshold} reached)"
                )
                if self.on_open_callback:
                    self.on_open_callback()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta una función protegida por el circuit breaker
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            CircuitBreakerOpenError: Si el circuit breaker está abierto
        """
        self.state_obj.total_requests += 1
        
        if not self._should_attempt_request():
            from aws.exceptions import CircuitBreakerOpenError
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable. Last failure: {self.state_obj.last_failure_time}"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise
        except Exception as e:
            # Otras excepciones también cuentan como fallos
            self._record_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Versión async de call"""
        self.state_obj.total_requests += 1
        
        if not self._should_attempt_request():
            from aws.exceptions import CircuitBreakerOpenError
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable. Last failure: {self.state_obj.last_failure_time}"
            )
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise
        except Exception as e:
            self._record_failure()
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del circuit breaker"""
        return {
            "name": self.name,
            "state": self.state_obj.state.value,
            "failure_count": self.state_obj.failure_count,
            "success_count": self.state_obj.success_count,
            "total_requests": self.state_obj.total_requests,
            "total_failures": self.state_obj.total_failures,
            "total_successes": self.state_obj.total_successes,
            "last_failure_time": self.state_obj.last_failure_time,
            "last_success_time": self.state_obj.last_success_time,
            "failure_rate": (
                self.state_obj.total_failures / self.state_obj.total_requests
                if self.state_obj.total_requests > 0 else 0
            )
        }
    
    def reset(self):
        """Resetea el circuit breaker a estado CLOSED"""
        self.state_obj.state = CircuitState.CLOSED
        self.state_obj.failure_count = 0
        self.state_obj.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")


# Decorator para usar circuit breaker fácilmente
def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    on_open: Optional[Callable] = None,
    on_close: Optional[Callable] = None
):
    """
    Decorator para aplicar circuit breaker a una función
    
    Usage:
        @circuit_breaker("external_api", CircuitBreakerConfig(failure_threshold=3))
        def call_external_api():
            ...
    """
    breaker = CircuitBreaker(name, config, on_open, on_close)
    
    def decorator(func):
        if hasattr(func, '__call__') and hasattr(func, '__await__'):
            # Async function
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            # Sync function
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Circuit breakers globales para servicios comunes
_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Obtiene o crea un circuit breaker global"""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, config)
    return _breakers[name]
















"""
Circuit Breaker - Sistema de circuit breaker para resiliencia
=============================================================
"""

import logging
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"  # Normal, permitiendo requests
    OPEN = "open"  # Bloqueando requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


class CircuitBreaker:
    """Circuit breaker para proteger servicios"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60,
                 success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta una función con protección de circuit breaker"""
        if self.state == CircuitState.OPEN:
            # Verificar si debemos intentar half-open
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker: Cambiando a HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta una función async con protección de circuit breaker"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker: Cambiando a HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Maneja éxito"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker: Cambiando a CLOSED (servicio recuperado)")
    
    def _on_failure(self):
        """Maneja fallo"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Fallo en half-open, volver a open
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.warning("Circuit breaker: Cambiando a OPEN (fallo en half-open)")
        elif self.failure_count >= self.failure_threshold:
            # Demasiados fallos, abrir circuit
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()
                logger.warning(f"Circuit breaker: Cambiando a OPEN ({self.failure_count} fallos)")
    
    def _should_attempt_reset(self) -> bool:
        """Verifica si debemos intentar resetear"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout_seconds
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado del circuit breaker"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat()
        }
    
    def reset(self):
        """Resetea el circuit breaker manualmente"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()
        logger.info("Circuit breaker: Reset manual")


def circuit_breaker_decorator(breaker: CircuitBreaker):
    """Decorador para usar circuit breaker"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)
        return wrapper
    return decorator





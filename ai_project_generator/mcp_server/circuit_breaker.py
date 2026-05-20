"""
MCP Circuit Breaker - Circuit breaker pattern para resiliencia
==============================================================
"""

import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
from functools import wraps

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker para prevenir cascading failures
    
    Implementa el patrón circuit breaker:
    - CLOSED: Funciona normalmente
    - OPEN: Rechaza requests después de muchos fallos
    - HALF_OPEN: Prueba si el servicio se recuperó
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "circuit_breaker",
    ):
        """
        Args:
            failure_threshold: Número de fallos antes de abrir
            recovery_timeout: Tiempo en segundos antes de intentar recuperación
            expected_exception: Tipo de excepción que cuenta como fallo
            name: Nombre del circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self.half_open_attempts = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta función a través del circuit breaker
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la función
            
        Raises:
            MCPError: Si el circuit está abierto
            expected_exception: Si la función falla
        """
        # Verificar estado
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                raise MCPError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Service unavailable. Retry after {self.recovery_timeout}s"
                )
        
        # Ejecutar función
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta función async a través del circuit breaker
        
        Args:
            func: Función async a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la función
            
        Raises:
            MCPError: Si el circuit está abierto
            expected_exception: Si la función falla
        """
        # Verificar estado
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                raise MCPError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Service unavailable. Retry after {self.recovery_timeout}s"
                )
        
        # Ejecutar función
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Maneja éxito"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Requiere 2 éxitos consecutivos
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED state")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Maneja fallo"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            # Fallo en half-open, volver a open
            self.state = CircuitState.OPEN
            self.half_open_attempts = 0
            logger.warning(f"Circuit breaker {self.name} failed in HALF_OPEN, returning to OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker {self.name} opened after {self.failure_count} failures"
            )
    
    def _should_attempt_recovery(self) -> bool:
        """Verifica si se debe intentar recuperación"""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def get_state(self) -> dict:
        """Obtiene estado actual del circuit breaker"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "success_count": self.success_count,
            "half_open_attempts": self.half_open_attempts,
        }
    
    def reset(self):
        """Resetea el circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.half_open_attempts = 0
        logger.info(f"Circuit breaker {self.name} reset")


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    name: str = "circuit_breaker",
):
    """
    Decorador para aplicar circuit breaker a funciones
    
    Args:
        failure_threshold: Número de fallos antes de abrir
        recovery_timeout: Tiempo antes de intentar recuperación
        expected_exception: Tipo de excepción que cuenta como fallo
        name: Nombre del circuit breaker
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name=name,
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)
        return wrapper
    return decorator


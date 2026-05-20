"""
Sistema avanzado de retry con circuit breaker y backoff adaptativo.
"""

import asyncio
import time
from typing import Callable, TypeVar, Optional, Dict, Any, List
from enum import Enum
from functools import wraps
from datetime import datetime, timedelta

from config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"  # Normal, permitir requests
    OPEN = "open"  # Fallando, rechazar requests
    HALF_OPEN = "half_open"  # Probando si se recuperó


class CircuitBreaker:
    """Circuit breaker para prevenir cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        name: str = "default"
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            failure_threshold: Número de fallos antes de abrir
            success_threshold: Número de éxitos para cerrar desde half-open
            timeout: Tiempo en segundos antes de intentar half-open
            name: Nombre del circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_opens": 0
        }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar función con circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuit está abierto o la función falla
        """
        self.stats["total_requests"] += 1
        
        # Verificar estado
        if self.state == CircuitState.OPEN:
            # Verificar si debemos intentar half-open
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    logger.info(f"Circuit breaker {self.name}: OPEN -> HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Retry after {self.timeout - elapsed:.1f}s"
                    )
        
        # Ejecutar función
        try:
            result = func(*args, **kwargs)
            
            # Éxito
            self._on_success()
            return result
            
        except Exception as e:
            # Falla
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Versión async de call."""
        self.stats["total_requests"] += 1
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    logger.info(f"Circuit breaker {self.name}: OPEN -> HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Retry after {self.timeout - elapsed:.1f}s"
                    )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Manejar éxito."""
        self.stats["successful_requests"] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit breaker {self.name}: HALF_OPEN -> CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset contador de fallos
    
    def _on_failure(self) -> None:
        """Manejar fallo."""
        self.stats["failed_requests"] += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker {self.name}: HALF_OPEN -> OPEN")
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker {self.name}: CLOSED -> OPEN")
                self.state = CircuitState.OPEN
                self.stats["circuit_opens"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            **self.stats
        }


def retry_with_circuit_breaker(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    exceptions: tuple = (Exception,)
):
    """
    Decorador para retry con circuit breaker.
    
    Args:
        max_attempts: Número máximo de intentos
        min_wait: Tiempo mínimo de espera (segundos)
        max_wait: Tiempo máximo de espera (segundos)
        circuit_breaker: Circuit breaker a usar (opcional)
        exceptions: Excepciones que activan retry
        
    Returns:
        Decorador
    """
    if circuit_breaker is None:
        circuit_breaker = CircuitBreaker(name="default")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            wait_time = min_wait
            
            while attempt < max_attempts:
                attempt += 1
                
                try:
                    # Intentar con circuit breaker
                    if asyncio.iscoroutinefunction(func):
                        return await circuit_breaker.call_async(func, *args, **kwargs)
                    else:
                        return circuit_breaker.call(func, *args, **kwargs)
                        
                except Exception as e:
                    # Verificar si debemos retry
                    if not isinstance(e, exceptions):
                        raise  # No retry para excepciones no esperadas
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"Max attempts ({max_attempts}) reached for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    # Esperar con backoff exponencial
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait)
            
            raise Exception(f"Max attempts reached for {func.__name__}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            wait_time = min_wait
            
            while attempt < max_attempts:
                attempt += 1
                
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    if not isinstance(e, exceptions):
                        raise
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"Max attempts ({max_attempts}) reached for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait)
            
            raise Exception(f"Max attempts reached for {func.__name__}")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator




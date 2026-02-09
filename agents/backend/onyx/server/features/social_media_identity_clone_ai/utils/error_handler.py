"""
Manejo robusto de errores con retry y circuit breaker
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Type, Tuple, List
from enum import Enum
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Tipos de errores"""
    NETWORK = "network"
    API = "api"
    VALIDATION = "validation"
    PROCESSING = "processing"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuración de retry"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retryable_errors: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
        asyncio.TimeoutError,
    )
    non_retryable_errors: Tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
    )


@dataclass
class CircuitBreakerConfig:
    """Configuración de circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker para prevenir llamadas a servicios fallidos"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función con circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_calls = 0
                logger.info("Circuit breaker: transitioning to half-open")
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
        """Ejecuta función async con circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_calls = 0
                logger.info("Circuit breaker: transitioning to half-open")
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
        if self.state == "half_open":
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker: CLOSED (recovered)")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Maneja fallo"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "half_open":
            self.state = "open"
            logger.warning("Circuit breaker: OPEN (half-open failed)")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker: OPEN (threshold reached: {self.failure_count})")
    
    def _should_attempt_reset(self) -> bool:
        """Verifica si se debe intentar reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout


class RetryHandler:
    """Maneja retries con exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función con retry"""
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if not self._should_retry(e, attempt):
                    raise e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Retry attempt {attempt + 1}/{self.config.max_attempts} after {delay:.2f}s: {e}")
                    time.sleep(delay)
        
        raise last_error or Exception("Unknown error in retry handler")
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función async con retry"""
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if not self._should_retry(e, attempt):
                    raise e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Retry attempt {attempt + 1}/{self.config.max_attempts} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
        
        raise last_error or Exception("Unknown error in retry handler")
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determina si se debe hacer retry"""
        # No retry si es el último intento
        if attempt >= self.config.max_attempts - 1:
            return False
        
        # No retry para errores no retryables
        if any(isinstance(error, error_type) for error_type in self.config.non_retryable_errors):
            return False
        
        # Retry para errores retryables
        if any(isinstance(error, error_type) for error_type in self.config.retryable_errors):
            return True
        
        # Por defecto, no retry
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calcula delay con exponential backoff"""
        delay = min(
            self.config.base_delay * (self.config.backoff_factor ** attempt),
            self.config.max_delay
        )
        return delay


def retry_on_error(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None
):
    """Decorator para retry automático"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                retryable_errors=retryable_errors or RetryConfig().retryable_errors
            )
            handler = RetryHandler(config)
            return await handler.execute_async(func, *args, **kwargs)
        return wrapper
    return decorator





"""
Resilient Operations Module
===========================

Operaciones resilientes con retries, timeouts y circuit breakers.
Proporciona patrones comunes para operaciones robustas.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Type, Tuple, Dict
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta

from .error_handler import AgentError, ErrorSeverity

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker para prevenir llamadas a servicios fallidos.
    
    Implementa el patrón Circuit Breaker para proteger contra
    fallos en cascada.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            failure_threshold: Número de fallos antes de abrir
            recovery_timeout: Tiempo antes de intentar recuperación
            expected_exception: Tipo de excepción que cuenta como fallo
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
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
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise AgentError(
                    "Circuit breaker is OPEN",
                    severity=ErrorSeverity.HIGH,
                    context={"state": self.state.value}
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Éxito: resetear contador
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker recovered, entering CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar función async con circuit breaker.
        
        Args:
            func: Función async a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuit está abierto o la función falla
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise AgentError(
                    "Circuit breaker is OPEN",
                    severity=ErrorSeverity.HIGH,
                    context={"state": self.state.value}
                )
        
        try:
            result = await func(*args, **kwargs)
            
            # Éxito: resetear contador
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker recovered, entering CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise
    
    def _record_failure(self) -> None:
        """Registrar un fallo."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Verificar si se debe intentar resetear."""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def reset(self) -> None:
        """Resetear circuit breaker manualmente."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


def resilient_call(
    func: Callable,
    *args,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    backoff: float = 1.0,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs
) -> Any:
    """
    Ejecutar función con resiliencia (retries, timeout, circuit breaker).
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        max_retries: Número máximo de reintentos
        timeout: Timeout en segundos
        backoff: Factor de espera exponencial
        retryable_errors: Tipos de errores que se pueden reintentar
        circuit_breaker: Circuit breaker opcional
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la función
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Aplicar circuit breaker si está disponible
            if circuit_breaker:
                if asyncio.iscoroutinefunction(func):
                    result = await circuit_breaker.call_async(func, *args, **kwargs)
                else:
                    result = circuit_breaker.call(func, *args, **kwargs)
            else:
                # Aplicar timeout si está disponible
                if timeout:
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout
                        )
                    else:
                        # Para funciones síncronas, usar threading
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, *args, **kwargs)
                            result = future.result(timeout=timeout)
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            last_error = e
            
            # Verificar si el error es retryable
            if retryable_errors and not isinstance(e, retryable_errors):
                raise
            
            # Si no hay más intentos, lanzar error
            if attempt >= max_retries:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}"
                )
                raise
            
            # Esperar antes de reintentar
            wait_time = backoff * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}. "
                f"Retrying in {wait_time}s..."
            )
            
            if asyncio.iscoroutinefunction(func):
                await asyncio.sleep(wait_time)
            else:
                time.sleep(wait_time)
    
    raise last_error


async def resilient_call_async(
    func: Callable,
    *args,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    backoff: float = 1.0,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs
) -> Any:
    """
    Ejecutar función async con resiliencia.
    
    Args:
        func: Función async a ejecutar
        *args: Argumentos posicionales
        max_retries: Número máximo de reintentos
        timeout: Timeout en segundos
        backoff: Factor de espera exponencial
        retryable_errors: Tipos de errores que se pueden reintentar
        circuit_breaker: Circuit breaker opcional
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la función
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Aplicar circuit breaker si está disponible
            if circuit_breaker:
                result = await circuit_breaker.call_async(func, *args, **kwargs)
            else:
                # Aplicar timeout si está disponible
                if timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            last_error = e
            
            # Verificar si el error es retryable
            if retryable_errors and not isinstance(e, retryable_errors):
                raise
            
            # Si no hay más intentos, lanzar error
            if attempt >= max_retries:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}"
                )
                raise
            
            # Esperar antes de reintentar
            wait_time = backoff * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}. "
                f"Retrying in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)
    
    raise last_error


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
) -> CircuitBreaker:
    """
    Factory function para crear CircuitBreaker.
    
    Args:
        failure_threshold: Número de fallos antes de abrir
        recovery_timeout: Tiempo antes de intentar recuperación
        expected_exception: Tipo de excepción que cuenta como fallo
        
    Returns:
        Instancia de CircuitBreaker
    """
    return CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)



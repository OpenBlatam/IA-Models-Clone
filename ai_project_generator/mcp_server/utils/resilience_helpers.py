"""
Resilience Helpers - Utilidades para mejorar la resiliencia del servidor
========================================================================

Funciones helper para mejorar la resiliencia y recuperación de errores.
"""

import logging
import asyncio
import time
from typing import Callable, Any, Optional, TypeVar, List, Dict, Type
from functools import wraps
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from ..exceptions import MCPError, MCPConnectorError, MCPOperationError
from ..retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorRecoveryStrategy:
    """
    Estrategia de recuperación de errores.
    
    Define cómo recuperarse de diferentes tipos de errores.
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        fallback_value: Any = None,
        fallback_func: Optional[Callable] = None,
        circuit_breaker: Optional[Any] = None
    ):
        """
        Inicializar estrategia de recuperación.
        
        Args:
            retry_config: Configuración de reintentos (opcional)
            fallback_value: Valor de fallback si todo falla (opcional)
            fallback_func: Función de fallback (opcional)
            circuit_breaker: Circuit breaker (opcional)
        """
        self.retry_config = retry_config or RetryConfig(max_attempts=3)
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
        self.circuit_breaker = circuit_breaker
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con estrategia de recuperación.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado de la función o fallback
        """
        # Intentar con circuit breaker si está disponible
        if self.circuit_breaker:
            try:
                return await self.circuit_breaker.call(func, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Circuit breaker triggered: {e}")
                return self._get_fallback()
        
        # Intentar con retry
        try:
            return await retry_with_backoff(
                func,
                *args,
                config=self.retry_config,
                **kwargs
            )
        except Exception as e:
            logger.error(f"All retry attempts failed: {e}")
            
            # Intentar fallback function
            if self.fallback_func:
                try:
                    return await self.fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback function also failed: {fallback_error}")
            
            # Retornar fallback value
            return self._get_fallback()
    
    def _get_fallback(self) -> Any:
        """Obtener valor de fallback"""
        if self.fallback_value is not None:
            return self.fallback_value
        raise ValueError("No fallback available and all attempts failed")


class TimeoutHandler:
    """
    Handler para timeouts de operaciones.
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        Inicializar handler de timeout.
        
        Args:
            timeout: Timeout en segundos (default: 30.0)
        """
        self.timeout = timeout
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con timeout.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado de la función
        
        Raises:
            asyncio.TimeoutError: Si la función excede el timeout
        """
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {self.timeout}s")
            raise


class Bulkhead:
    """
    Bulkhead pattern para limitar concurrencia.
    
    Limita el número de operaciones concurrentes para proteger recursos.
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Inicializar bulkhead.
        
        Args:
            max_concurrent: Número máximo de operaciones concurrentes
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con limitación de concurrencia.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado de la función
        """
        async with self.semaphore:
            self.active_count += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self.active_count -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del bulkhead.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "max_concurrent": self.max_concurrent,
            "active": self.active_count,
            "available": self.max_concurrent - self.active_count
        }


class HealthCheck:
    """
    Health check para componentes.
    """
    
    def __init__(
        self,
        check_func: Callable,
        interval: float = 60.0,
        timeout: float = 5.0,
        failure_threshold: int = 3
    ):
        """
        Inicializar health check.
        
        Args:
            check_func: Función de health check
            interval: Intervalo entre checks en segundos
            timeout: Timeout para cada check
            failure_threshold: Número de fallos antes de marcar como unhealthy
        """
        self.check_func = check_func
        self.interval = interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.last_check: Optional[datetime] = None
        self.last_result: Optional[bool] = None
        self._running = False
    
    async def check(self) -> bool:
        """
        Ejecutar health check.
        
        Returns:
            True si está saludable, False si no
        """
        try:
            result = await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout
            )
            self.last_result = bool(result)
            self.last_check = datetime.utcnow()
            
            if self.last_result:
                self.failure_count = 0
            else:
                self.failure_count += 1
            
            return self.last_result
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.failure_count += 1
            self.last_result = False
            self.last_check = datetime.utcnow()
            return False
    
    def is_healthy(self) -> bool:
        """
        Verificar si el componente está saludable.
        
        Returns:
            True si está saludable
        """
        return (
            self.last_result is True and
            self.failure_count < self.failure_threshold
        )
    
    async def start_monitoring(self) -> None:
        """Iniciar monitoreo continuo"""
        self._running = True
        while self._running:
            await self.check()
            await asyncio.sleep(self.interval)
    
    def stop_monitoring(self) -> None:
        """Detener monitoreo"""
        self._running = False


def with_error_recovery(
    retry_config: Optional[RetryConfig] = None,
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None,
    timeout: Optional[float] = None
):
    """
    Decorator para agregar recuperación de errores a funciones.
    
    Args:
        retry_config: Configuración de reintentos
        fallback_value: Valor de fallback
        fallback_func: Función de fallback
        timeout: Timeout en segundos
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        strategy = ErrorRecoveryStrategy(
            retry_config=retry_config,
            fallback_value=fallback_value,
            fallback_func=fallback_func
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if timeout:
                timeout_handler = TimeoutHandler(timeout=timeout)
                return await timeout_handler.execute(
                    strategy.execute,
                    func,
                    *args,
                    **kwargs
                )
            return await strategy.execute(func, *args, **kwargs)
        
        return wrapper
    return decorator


def with_bulkhead(max_concurrent: int = 10):
    """
    Decorator para agregar bulkhead a funciones.
    
    Args:
        max_concurrent: Número máximo de operaciones concurrentes
    
    Returns:
        Decorator function
    """
    bulkhead = Bulkhead(max_concurrent=max_concurrent)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await bulkhead.execute(func, *args, **kwargs)
        return wrapper
    return decorator


@asynccontextmanager
async def resilient_operation(
    operation_name: str,
    retry_config: Optional[RetryConfig] = None,
    timeout: Optional[float] = None,
    fallback_value: Any = None
):
    """
    Context manager para operaciones resilientes.
    
    Args:
        operation_name: Nombre de la operación
        retry_config: Configuración de reintentos
        timeout: Timeout en segundos
        fallback_value: Valor de fallback
    
    Yields:
        None
    """
    start_time = time.time()
    strategy = ErrorRecoveryStrategy(
        retry_config=retry_config,
        fallback_value=fallback_value
    )
    
    try:
        yield strategy
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Operation {operation_name} failed after {duration:.2f}s: {e}",
            exc_info=True
        )
        raise
    finally:
        duration = time.time() - start_time
        logger.debug(f"Operation {operation_name} completed in {duration:.2f}s")


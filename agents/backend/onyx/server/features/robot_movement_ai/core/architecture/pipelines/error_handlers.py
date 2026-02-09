"""
Pipeline Error Handlers
=======================

Manejadores de errores para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, TypeVar, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .stages import PipelineStage

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class ErrorHandlerConfig:
    """Configuración para manejadores de errores."""
    max_retries: int = 3
    retry_delay: float = 1.0
    failure_threshold: int = 5
    timeout: float = 60.0
    half_open_timeout: float = 30.0


class ErrorHandler(ABC):
    """
    Interfaz base para manejadores de errores.
    """
    
    @abstractmethod
    async def handle(
        self,
        error: Exception,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Manejar error.

        Args:
            error: Excepción
            stage: Etapa que falló
            data: Datos originales
            context: Contexto

        Returns:
            Datos procesados o valor por defecto
        """
        pass


class DefaultErrorHandler(ErrorHandler):
    """
    Manejador de errores por defecto (re-lanza excepción).
    """
    
    async def handle(
        self,
        error: Exception,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Re-lanzar excepción con logging.
        
        Args:
            error: Excepción
            stage: Etapa que falló
            data: Datos originales
            context: Contexto
            
        Raises:
            Exception: La excepción original
        """
        logger.error(
            f"Error in stage '{stage.get_name()}': {error}",
            exc_info=error
        )
        raise error


class RetryErrorHandler(ErrorHandler):
    """
    Manejador de errores con reintentos.
    Optimizado con async/await y mejor manejo de errores.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_exceptions: Tuple[type[Exception], ...] = (Exception,)
    ) -> None:
        """
        Inicializar manejador con reintentos.
        
        Args:
            max_retries: Número máximo de reintentos
            retry_delay: Delay entre reintentos en segundos
            retry_exceptions: Excepciones que activan reintento
        """
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exceptions = retry_exceptions
    
    async def handle(
        self,
        error: Exception,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Manejar con reintentos.
        
        Args:
            error: Excepción
            stage: Etapa que falló
            data: Datos originales
            context: Contexto
            
        Returns:
            Datos procesados
            
        Raises:
            Exception: Si todos los reintentos fallan
        """
        if not isinstance(error, self.retry_exceptions):
            raise error
        
        last_error = error
        
        for attempt in range(self.max_retries):
            logger.warning(
                f"Retrying stage '{stage.get_name()}' "
                f"(attempt {attempt + 1}/{self.max_retries})"
            )
            
            if self.retry_delay > 0:
                await asyncio.sleep(self.retry_delay)
            
            try:
                if hasattr(stage, 'execute'):
                    return await stage.execute(data, context)
                elif hasattr(stage, 'process'):
                    return stage.process(data, context)
                else:
                    raise ValueError(f"Stage '{stage.get_name()}' has no execute or process method")
            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Stage '{stage.get_name()}' failed after "
                        f"{self.max_retries} retries"
                    )
                    raise e
        
        raise last_error


class CircuitBreakerErrorHandler(ErrorHandler):
    """
    Manejador de errores con circuit breaker.
    Optimizado con mejor manejo de estados y async/await.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ) -> None:
        """
        Inicializar circuit breaker.
        
        Args:
            failure_threshold: Umbral de fallos
            timeout: Timeout en estado abierto (segundos)
            half_open_timeout: Timeout en estado half-open (segundos)
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        
        if half_open_timeout <= 0:
            raise ValueError("half_open_timeout must be positive")
        
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
    
    async def handle(
        self,
        error: Exception,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Manejar con circuit breaker.
        
        Args:
            error: Excepción
            stage: Etapa que falló
            data: Datos originales
            context: Contexto
            
        Returns:
            Datos procesados
            
        Raises:
            Exception: Si el circuit breaker está abierto o falla
        """
        import time
        
        async with self._lock:
            current_time = time.time()
            
            await self._update_state(current_time, stage)
            
            if self.state == CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker is OPEN for stage '{stage.get_name()}'"
                )
                raise error
            
            if self.state == CircuitState.HALF_OPEN:
                return await self._try_half_open(stage, data, context)
            
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker OPENED for stage '{stage.get_name()}' "
                    f"(failures: {self.failure_count})"
                )
            
            raise error
    
    async def _update_state(self, current_time: float, stage: PipelineStage) -> None:
        """Actualizar estado del circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               (current_time - self.last_failure_time) > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
                logger.info(
                    f"Circuit breaker HALF-OPEN for stage '{stage.get_name()}'"
                )
    
    async def _try_half_open(
        self,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]]
    ) -> T:
        """Intentar ejecución en estado half-open."""
        import time
        
        try:
            if hasattr(stage, 'execute'):
                result = await stage.execute(data, context)
            elif hasattr(stage, 'process'):
                result = stage.process(data, context)
            else:
                raise ValueError(
                    f"Stage '{stage.get_name()}' has no execute or process method"
                )
            
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info(
                f"Circuit breaker CLOSED for stage '{stage.get_name()}'"
            )
            return result
        except Exception as e:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            raise e

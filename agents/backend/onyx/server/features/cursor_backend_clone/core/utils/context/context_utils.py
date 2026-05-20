"""
Context Utils - Utilidades de Contexto Avanzadas
=================================================

Utilidades para manejo de contexto y context managers avanzados.
"""

import logging
import asyncio
import time
from typing import Any, Optional, Dict, Callable, ContextManager
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


@contextmanager
def timer_context(name: Optional[str] = None):
    """
    Context manager para medir tiempo de ejecución.
    
    Args:
        name: Nombre opcional para el timer
        
    Yields:
        TimerContext con elapsed_time
    """
    start = time.time()
    context = type('TimerContext', (), {'elapsed_time': 0.0})()
    
    try:
        yield context
    finally:
        elapsed = time.time() - start
        context.elapsed_time = elapsed
        if name:
            logger.debug(f"⏱️ {name} took {elapsed:.3f}s")
        else:
            logger.debug(f"⏱️ Operation took {elapsed:.3f}s")


@asynccontextmanager
async def async_timer_context(name: Optional[str] = None):
    """
    Context manager async para medir tiempo de ejecución.
    
    Args:
        name: Nombre opcional para el timer
        
    Yields:
        TimerContext con elapsed_time
    """
    start = time.time()
    context = type('TimerContext', (), {'elapsed_time': 0.0})()
    
    try:
        yield context
    finally:
        elapsed = time.time() - start
        context.elapsed_time = elapsed
        if name:
            logger.debug(f"⏱️ {name} took {elapsed:.3f}s")
        else:
            logger.debug(f"⏱️ Operation took {elapsed:.3f}s")


@contextmanager
def suppress_exceptions(*exception_types, default_return: Any = None):
    """
    Context manager para suprimir excepciones específicas.
    
    Args:
        *exception_types: Tipos de excepciones a suprimir
        default_return: Valor a retornar si se suprime excepción
        
    Yields:
        None
    """
    try:
        yield
    except exception_types as e:
        logger.debug(f"Suppressed exception: {type(e).__name__}: {e}")
        if default_return is not None:
            return default_return


@asynccontextmanager
async def async_suppress_exceptions(*exception_types, default_return: Any = None):
    """
    Context manager async para suprimir excepciones específicas.
    
    Args:
        *exception_types: Tipos de excepciones a suprimir
        default_return: Valor a retornar si se suprime excepción
        
    Yields:
        None
    """
    try:
        yield
    except exception_types as e:
        logger.debug(f"Suppressed exception: {type(e).__name__}: {e}")
        if default_return is not None:
            return default_return


@contextmanager
def context_variables(**variables):
    """
    Context manager para establecer variables de contexto.
    
    Args:
        **variables: Variables a establecer en contexto
        
    Yields:
        Diccionario con variables
    """
    context = variables.copy()
    try:
        yield context
    finally:
        pass  # Variables se limpian automáticamente


@asynccontextmanager
async def async_context_variables(**variables):
    """
    Context manager async para establecer variables de contexto.
    
    Args:
        **variables: Variables a establecer en contexto
        
    Yields:
        Diccionario con variables
    """
    context = variables.copy()
    try:
        yield context
    finally:
        pass


@contextmanager
def retry_context(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Context manager para reintentos automáticos.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre intentos
        backoff: Factor de backoff
        
    Yields:
        RetryContext con attempt_count
    """
    context = type('RetryContext', (), {'attempt_count': 0, 'last_exception': None})()
    
    for attempt in range(max_attempts):
        context.attempt_count = attempt + 1
        try:
            yield context
            return
        except Exception as e:
            context.last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (backoff ** attempt))
            else:
                raise


@asynccontextmanager
async def async_retry_context(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Context manager async para reintentos automáticos.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre intentos
        backoff: Factor de backoff
        
    Yields:
        RetryContext con attempt_count
    """
    context = type('RetryContext', (), {'attempt_count': 0, 'last_exception': None})()
    
    for attempt in range(max_attempts):
        context.attempt_count = attempt + 1
        try:
            yield context
            return
        except Exception as e:
            context.last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay * (backoff ** attempt))
            else:
                raise


@contextmanager
def timeout_context(seconds: float):
    """
    Context manager para timeout (sync).
    
    Args:
        seconds: Segundos de timeout
        
    Yields:
        None
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    # Configurar signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@asynccontextmanager
async def async_timeout_context(seconds: float):
    """
    Context manager async para timeout.
    
    Args:
        seconds: Segundos de timeout
        
    Yields:
        None
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds}s")


class ContextManager:
    """
    Context manager genérico con callbacks.
    """
    
    def __init__(
        self,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.on_error = on_error
    
    def __enter__(self):
        if self.on_enter:
            return self.on_enter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.on_error:
            self.on_error(exc_type, exc_val, exc_tb)
        
        if self.on_exit:
            self.on_exit(exc_type, exc_val, exc_tb)
        
        return False  # No suprimir excepciones


class AsyncContextManager:
    """
    Context manager async genérico con callbacks.
    """
    
    def __init__(
        self,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.on_error = on_error
    
    async def __aenter__(self):
        if self.on_enter:
            if asyncio.iscoroutinefunction(self.on_enter):
                return await self.on_enter()
            else:
                return self.on_enter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.on_error:
            if asyncio.iscoroutinefunction(self.on_error):
                await self.on_error(exc_type, exc_val, exc_tb)
            else:
                self.on_error(exc_type, exc_val, exc_tb)
        
        if self.on_exit:
            if asyncio.iscoroutinefunction(self.on_exit):
                await self.on_exit(exc_type, exc_val, exc_tb)
            else:
                self.on_exit(exc_type, exc_val, exc_tb)
        
        return False  # No suprimir excepciones





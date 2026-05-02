"""
Async Utilities Module
======================

Utilidades para operaciones asíncronas y manejo de concurrencia.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, List, TypeVar, Coroutine
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def run_with_timeout(
    coro: Coroutine,
    timeout: float,
    default: Any = None,
    timeout_handler: Optional[Callable] = None
) -> Any:
    """
    Ejecutar coroutine con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Tiempo máximo en segundos
        default: Valor por defecto si hay timeout
        timeout_handler: Handler a ejecutar en caso de timeout
        
    Returns:
        Resultado de la coroutine o default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        if timeout_handler:
            if asyncio.iscoroutinefunction(timeout_handler):
                await timeout_handler()
            else:
                timeout_handler()
        return default
    except Exception as e:
        logger.error(f"Error in async operation: {e}", exc_info=True)
        return default


async def run_with_retry(
    coro: Coroutine,
    max_retries: int = 3,
    backoff: float = 1.0,
    retryable_errors: Optional[tuple] = None
) -> Any:
    """
    Ejecutar coroutine con reintentos.
    
    Args:
        coro: Coroutine a ejecutar
        max_retries: Número máximo de reintentos
        backoff: Factor de espera exponencial
        retryable_errors: Tipos de errores que se pueden reintentar
        
    Returns:
        Resultado de la coroutine
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return await coro
        except Exception as e:
            last_error = e
            
            if retryable_errors and not isinstance(e, retryable_errors):
                raise
            
            if attempt < max_retries:
                wait_time = backoff * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
                raise
    
    raise last_error


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int = 5,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Ejecutar múltiples coroutines con límite de concurrencia.
    
    Args:
        coros: Lista de coroutines
        limit: Límite de concurrencia
        return_exceptions: Si se deben retornar excepciones en lugar de lanzarlas
        
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro
    
    tasks = [run_with_semaphore(coro) for coro in coros]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


async def run_periodically(
    func: Callable,
    interval: float,
    stop_event: Optional[asyncio.Event] = None,
    initial_delay: float = 0.0
) -> None:
    """
    Ejecutar función periódicamente.
    
    Args:
        func: Función a ejecutar
        interval: Intervalo entre ejecuciones
        stop_event: Evento para detener la ejecución
        initial_delay: Retraso inicial antes de la primera ejecución
    """
    if initial_delay > 0:
        await asyncio.sleep(initial_delay)
    
    stop = stop_event or asyncio.Event()
    
    while not stop.is_set():
        try:
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
        except Exception as e:
            logger.error(f"Error in periodic function: {e}", exc_info=True)
        
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue  # Continuar con el siguiente ciclo


class AsyncTaskManager:
    """Manager para tareas asíncronas con tracking."""
    
    def __init__(self):
        self.tasks: List[asyncio.Task] = []
        self.completed_tasks: List[asyncio.Task] = []
    
    def create_task(self, coro: Coroutine, name: Optional[str] = None) -> asyncio.Task:
        """
        Crear y trackear una tarea.
        
        Args:
            coro: Coroutine a ejecutar
            name: Nombre de la tarea
            
        Returns:
            Task creada
        """
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.tasks.append(task)
        return task
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> None:
        """
        Esperar a que todas las tareas se completen.
        
        Args:
            timeout: Tiempo máximo de espera
        """
        if not self.tasks:
            return
        
        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=timeout
                )
            else:
                await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for {len(self.tasks)} tasks")
    
    def cancel_all(self) -> None:
        """Cancelar todas las tareas."""
        for task in self.tasks:
            if not task.done():
                task.cancel()
    
    def cleanup_completed(self) -> None:
        """Limpiar tareas completadas."""
        self.completed_tasks = [t for t in self.tasks if t.done()]
        self.tasks = [t for t in self.tasks if not t.done()]
    
    def get_active_count(self) -> int:
        """Obtener número de tareas activas."""
        return len([t for t in self.tasks if not t.done()])


def async_retry(max_retries: int = 3, backoff: float = 1.0):
    """
    Decorador para reintentar funciones async.
    
    Args:
        max_retries: Número máximo de reintentos
        backoff: Factor de espera exponencial
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await run_with_retry(
                func(*args, **kwargs),
                max_retries=max_retries,
                backoff=backoff
            )
        return wrapper
    return decorator


def async_timeout(timeout: float, default: Any = None):
    """
    Decorador para agregar timeout a funciones async.
    
    Args:
        timeout: Tiempo máximo en segundos
        default: Valor por defecto si hay timeout
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await run_with_timeout(
                func(*args, **kwargs),
                timeout=timeout,
                default=default
            )
        return wrapper
    return decorator

"""
Async Utilities
===============

Utilidades para operaciones asíncronas.
"""

import asyncio
from typing import Any, Callable, Coroutine, List, Optional, TypeVar, Union
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Ejecutar coroutine con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        default: Valor por defecto si hay timeout
    
    Returns:
        Resultado de la coroutine o default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default
    except Exception as e:
        logger.error(f"Error in async operation: {e}", exc_info=True)
        return default


async def gather_with_limit(
    coros: List[Coroutine[Any, Any, T]],
    limit: int = 10
) -> List[T]:
    """
    Ejecutar múltiples coroutines con límite de concurrencia.
    
    Args:
        coros: Lista de coroutines
        limit: Límite de concurrencia
    
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[bounded_coro(coro) for coro in coros])


async def retry_async(
    func: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    Reintentar operación async con backoff exponencial.
    
    Args:
        func: Función async a ejecutar
        max_retries: Número máximo de reintentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff
        exceptions: Tipos de excepciones a capturar
    
    Returns:
        Resultado de la función
    
    Raises:
        Última excepción si todos los reintentos fallan
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception


def async_to_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Ejecutar coroutine de forma síncrona.
    
    Args:
        coro: Coroutine a ejecutar
    
    Returns:
        Resultado de la coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convertir función síncrona a async.
    
    Args:
        func: Función síncrona
    
    Returns:
        Función async
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper


class AsyncLock:
    """Lock asíncrono con contexto."""
    
    def __init__(self):
        """Inicializar lock."""
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Entrar al contexto."""
        await self._lock.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto."""
        self._lock.release()


class AsyncQueue:
    """Cola asíncrona con límite de tamaño."""
    
    def __init__(self, maxsize: int = 0):
        """
        Inicializar cola.
        
        Args:
            maxsize: Tamaño máximo (0 = ilimitado)
        """
        self._queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, item: Any) -> None:
        """Agregar item a la cola."""
        await self._queue.put(item)
    
    async def get(self) -> Any:
        """Obtener item de la cola."""
        return await self._queue.get()
    
    def qsize(self) -> int:
        """Obtener tamaño de la cola."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Verificar si la cola está vacía."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Verificar si la cola está llena."""
        return self._queue.full()


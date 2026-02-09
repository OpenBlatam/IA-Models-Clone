"""
Async Utilities - Utilidades asíncronas
========================================

Utilidades para trabajar con operaciones asíncronas.
"""

import asyncio
from typing import TypeVar, Callable, List, Any, Optional, Coroutine
from functools import wraps
import time

T = TypeVar('T')


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Ejecutar coroutines con límite de concurrencia.
    
    Args:
        coros: Lista de coroutines a ejecutar
        limit: Límite de concurrencia
        return_exceptions: Si True, las excepciones se retornan en lugar de lanzarse
    
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Coroutine):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *[bounded_coro(coro) for coro in coros],
        return_exceptions=return_exceptions
    )


async def timeout_after(
    coro: Coroutine[T, Any, T],
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
        Resultado o valor por defecto
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


def async_to_sync(func: Callable):
    """
    Convertir función async a sync.
    
    Args:
        func: Función async
    
    Returns:
        Función sync
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Cannot run async function in running event loop")
        return loop.run_until_complete(func(*args, **kwargs))
    
    return wrapper


def sync_to_async(func: Callable):
    """
    Convertir función sync a async.
    
    Args:
        func: Función sync
    
    Returns:
        Función async
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper


class AsyncRateLimiter:
    """Rate limiter asíncrono."""
    
    def __init__(self, max_calls: int, period: float):
        """
        Inicializar rate limiter.
        
        Args:
            max_calls: Número máximo de llamadas
            period: Período en segundos
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Adquirir permiso para ejecutar."""
        async with self._lock:
            now = time.time()
            self.calls = [call_time for call_time in self.calls if now - call_time < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    self.calls = []
            
            self.calls.append(time.time())
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


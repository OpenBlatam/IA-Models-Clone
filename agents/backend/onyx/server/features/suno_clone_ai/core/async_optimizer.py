"""
Async Optimizer
Optimizaciones para código asíncrono
"""

import logging
import asyncio
from typing import List, Any, Callable, Awaitable, TypeVar, Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncOptimizer:
    """Optimizador para código asíncrono"""
    
    def __init__(self):
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._rate_limiters: Dict[str, Any] = {}
    
    def get_semaphore(self, name: str, limit: int = 10) -> asyncio.Semaphore:
        """Obtiene o crea un semáforo"""
        if name not in self._semaphores:
            self._semaphores[name] = asyncio.Semaphore(limit)
        return self._semaphores[name]
    
    async def bounded_execute(
        self,
        coro: Awaitable[T],
        semaphore: asyncio.Semaphore
    ) -> T:
        """Ejecuta coroutine con límite de concurrencia"""
        async with semaphore:
            return await coro
    
    async def gather_with_limit(
        self,
        coros: List[Awaitable[T]],
        limit: int = 10
    ) -> List[T]:
        """
        Ejecuta coroutines con límite de concurrencia
        
        Args:
            coros: Lista de coroutines
            limit: Límite de concurrencia
            
        Returns:
            Lista de resultados
        """
        semaphore = asyncio.Semaphore(limit)
        
        async def bounded(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(*[bounded(coro) for coro in coros])
    
    async def timeout_execute(
        self,
        coro: Awaitable[T],
        timeout: float
    ) -> T:
        """
        Ejecuta coroutine con timeout
        
        Args:
            coro: Coroutine a ejecutar
            timeout: Timeout en segundos
            
        Returns:
            Resultado de la coroutine
        """
        return await asyncio.wait_for(coro, timeout=timeout)
    
    def create_task_pool(self, max_tasks: int = 100):
        """Crea pool de tareas"""
        return asyncio.Semaphore(max_tasks)


def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator para retry en funciones async"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        await asyncio.sleep(delay * attempt)
                    else:
                        raise last_exception
            return None
        return wrapper
    return decorator


def async_cache(ttl: int = 3600):
    """Decorator para cache en funciones async"""
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import time
            import hashlib
            
            # Generar key
            key = hashlib.md5(
                str(args) + str(sorted(kwargs.items())).encode()
            ).hexdigest()
            
            # Verificar cache
            if key in cache:
                if time.time() - cache_times[key] < ttl:
                    return cache[key]
                else:
                    del cache[key]
                    del cache_times[key]
            
            # Ejecutar y cachear
            result = await func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        return wrapper
    return decorator


# Instancia global
_async_optimizer: Optional[AsyncOptimizer] = None


def get_async_optimizer() -> AsyncOptimizer:
    """Obtiene el optimizador async"""
    global _async_optimizer
    if _async_optimizer is None:
        _async_optimizer = AsyncOptimizer()
    return _async_optimizer


"""
Async Utilities
================

Utilidades para programación asíncrona.
"""

import asyncio
from typing import List, Callable, Any, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


class AsyncBatchProcessor:
    """
    Procesador asíncrono de lotes.
    
    Procesa items en lotes de forma asíncrona.
    """
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        """
        Inicializar procesador.
        
        Args:
            batch_size: Tamaño del lote
            max_workers: Número máximo de workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable[[Any], Any]
    ) -> List[Any]:
        """
        Procesar items en lotes.
        
        Args:
            items: Lista de items
            processor: Función de procesamiento
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Procesar lote en paralelo
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                self.executor,
                lambda: [processor(item) for item in batch]
            )
            
            results.extend(batch_results)
        
        return results
    
    async def process_async(
        self,
        items: List[Any],
        processor: Callable[[Any], Coroutine]
    ) -> List[Any]:
        """
        Procesar items con función async.
        
        Args:
            items: Lista de items
            processor: Función async de procesamiento
            
        Returns:
            Lista de resultados
        """
        tasks = [processor(item) for item in items]
        return await asyncio.gather(*tasks)
    
    def shutdown(self):
        """Cerrar procesador."""
        self.executor.shutdown(wait=True)


async def run_in_background(coro: Coroutine) -> asyncio.Task:
    """
    Ejecutar coroutine en background.
    
    Args:
        coro: Coroutine a ejecutar
        
    Returns:
        Task creado
    """
    return asyncio.create_task(coro)


async def timeout_after(seconds: float, coro: Coroutine) -> Any:
    """
    Ejecutar coroutine con timeout.
    
    Args:
        seconds: Segundos de timeout
        coro: Coroutine a ejecutar
        
    Returns:
        Resultado de la coroutine
        
    Raises:
        asyncio.TimeoutError: Si excede el timeout
    """
    return await asyncio.wait_for(coro, timeout=seconds)


async def retry_async(
    func: Callable[[], Coroutine],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """
    Reintentar función async en caso de fallo.
    
    Args:
        func: Función async a ejecutar
        max_retries: Número máximo de reintentos
        delay: Delay inicial entre reintentos
        backoff: Factor de backoff exponencial
        
    Returns:
        Resultado de la función
        
    Raises:
        Exception: Si todos los reintentos fallan
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = delay * (backoff ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
    
    raise last_exception


class AsyncQueue:
    """
    Cola asíncrona con límite de tamaño.
    """
    
    def __init__(self, maxsize: int = 100):
        """
        Inicializar cola.
        
        Args:
            maxsize: Tamaño máximo
        """
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, item: Any) -> None:
        """Agregar item a la cola."""
        await self.queue.put(item)
    
    async def get(self) -> Any:
        """Obtener item de la cola."""
        return await self.queue.get()
    
    async def get_nowait(self) -> Any:
        """Obtener item sin esperar."""
        return self.queue.get_nowait()
    
    def qsize(self) -> int:
        """Obtener tamaño de la cola."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Verificar si la cola está vacía."""
        return self.queue.empty()
    
    def full(self) -> bool:
        """Verificar si la cola está llena."""
        return self.queue.full()


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int = 10
) -> List[Any]:
    """
    Ejecutar coroutines con límite de concurrencia.
    
    Args:
        coros: Lista de coroutines
        limit: Límite de concurrencia
        
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Coroutine) -> Any:
        async with semaphore:
            return await coro
    
    tasks = [bounded_coro(coro) for coro in coros]
    return await asyncio.gather(*tasks)


class AsyncLock:
    """
    Lock asíncrono con timeout.
    """
    
    def __init__(self):
        """Inicializar lock."""
        self.lock = asyncio.Lock()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Adquirir lock con timeout opcional.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            True si se adquirió el lock
        """
        if timeout is None:
            await self.lock.acquire()
            return True
        
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def release(self) -> None:
        """Liberar lock."""
        self.lock.release()
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.lock.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.lock.release()







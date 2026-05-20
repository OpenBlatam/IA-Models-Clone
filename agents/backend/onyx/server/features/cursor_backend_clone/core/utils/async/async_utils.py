"""
Async Utils - Utilidades Async Avanzadas
=========================================

Utilidades avanzadas para programación asíncrona.
"""

import logging
import asyncio
from typing import List, Any, Callable, Optional, Dict, Tuple
from collections import deque

logger = logging.getLogger(__name__)


async def gather_with_limit(
    tasks: List[Any],
    limit: int,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Ejecutar tareas con límite de concurrencia.
    
    Args:
        tasks: Lista de coroutines o tasks
        limit: Límite de concurrencia
        return_exceptions: Si retornar excepciones en lugar de lanzarlas
        
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *[bounded_task(task) for task in tasks],
        return_exceptions=return_exceptions
    )


async def gather_with_timeout(
    tasks: List[Any],
    timeout: float,
    return_exceptions: bool = False
) -> Tuple[List[Any], List[bool]]:
    """
    Ejecutar tareas con timeout global.
    
    Args:
        tasks: Lista de coroutines o tasks
        timeout: Timeout en segundos
        return_exceptions: Si retornar excepciones
        
    Returns:
        Tupla (resultados, completados)
    """
    try:
        async with asyncio.timeout(timeout):
            results = await asyncio.gather(
                *tasks,
                return_exceptions=return_exceptions
            )
            return results, [True] * len(results)
    except asyncio.TimeoutError:
        # Cancelar tareas pendientes
        for task in tasks:
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
        
        return [], [False] * len(tasks)


async def race(*tasks: Any) -> Any:
    """
    Ejecutar tareas y retornar el primer resultado.
    
    Args:
        *tasks: Tareas a ejecutar
        
    Returns:
        Resultado de la primera tarea completada
    """
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Cancelar tareas pendientes
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Retornar resultado de la primera completada
    return await done.pop()


async def retry_async(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Reintentar función async con backoff exponencial.
    
    Args:
        func: Función async a ejecutar
        max_attempts: Número máximo de intentos
        delay: Delay inicial
        backoff: Factor de backoff
        exceptions: Tipos de excepciones a capturar
        
    Returns:
        Resultado de la función
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = delay * (backoff ** attempt)
                logger.debug(f"Attempt {attempt + 1}/{max_attempts} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    if last_exception:
        raise last_exception


class AsyncQueue:
    """
    Cola async con límite de tamaño.
    """
    
    def __init__(self, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize
    
    async def put(self, item: Any) -> None:
        """Agregar item a la cola"""
        await self._queue.put(item)
    
    async def get(self) -> Any:
        """Obtener item de la cola"""
        return await self._queue.get()
    
    def qsize(self) -> int:
        """Obtener tamaño de la cola"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Verificar si está vacía"""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Verificar si está llena"""
        return self._queue.full()


class AsyncPool:
    """
    Pool de workers async para procesar items.
    """
    
    def __init__(self, size: int, worker: Callable):
        self.size = size
        self.worker = worker
        self.queue = AsyncQueue()
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    async def start(self) -> None:
        """Iniciar pool"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker_loop())
            for _ in range(self.size)
        ]
        logger.info(f"🚀 AsyncPool started with {self.size} workers")
    
    async def stop(self) -> None:
        """Detener pool"""
        self.running = False
        
        # Agregar None para cada worker (señal de parada)
        for _ in range(self.size):
            await self.queue.put(None)
        
        # Esperar a que terminen
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("🛑 AsyncPool stopped")
    
    async def submit(self, item: Any) -> None:
        """Enviar item para procesar"""
        await self.queue.put(item)
    
    async def _worker_loop(self) -> None:
        """Loop del worker"""
        while self.running:
            item = await self.queue.get()
            
            if item is None:  # Señal de parada
                break
            
            try:
                if asyncio.iscoroutinefunction(self.worker):
                    await self.worker(item)
                else:
                    self.worker(item)
            except Exception as e:
                logger.error(f"Error in worker: {e}")


async def batch_process_async(
    items: List[Any],
    processor: Callable,
    batch_size: int = 10,
    concurrency: int = 5
) -> List[Any]:
    """
    Procesar items en batches con concurrencia.
    
    Args:
        items: Lista de items a procesar
        processor: Función procesadora (async)
        batch_size: Tamaño de cada batch
        concurrency: Concurrencia máxima
        
    Returns:
        Lista de resultados
    """
    results = []
    
    # Dividir en batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Procesar batches con límite de concurrencia
    async def process_batch(batch):
        return await asyncio.gather(*[processor(item) for item in batch])
    
    batch_results = await gather_with_limit(
        [process_batch(batch) for batch in batches],
        limit=concurrency
    )
    
    # Aplanar resultados
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    return results


async def wait_for_first(
    *tasks: Any,
    timeout: Optional[float] = None
) -> Any:
    """
    Esperar a que la primera tarea complete.
    
    Args:
        *tasks: Tareas a esperar
        timeout: Timeout opcional
        
    Returns:
        Resultado de la primera tarea completada
    """
    if timeout:
        try:
            async with asyncio.timeout(timeout):
                return await race(*tasks)
        except asyncio.TimeoutError:
            raise TimeoutError(f"No task completed within {timeout}s")
    else:
        return await race(*tasks)





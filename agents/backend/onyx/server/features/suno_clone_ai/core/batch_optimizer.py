"""
Batch Optimizer
Optimización de operaciones en batch
"""

import logging
import asyncio
from typing import List, Any, Callable, TypeVar, Awaitable
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchProcessor:
    """
    Procesador de batches optimizado
    Agrupa operaciones para mejor rendimiento
    """
    
    def __init__(self, batch_size: int = 100, max_wait: float = 0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self._queue: deque = deque()
        self._pending_tasks: List[asyncio.Task] = []
        self._processing = False
    
    async def add(self, item: T) -> T:
        """Agrega item al batch"""
        self._queue.append(item)
        
        # Procesar si alcanzó el tamaño del batch
        if len(self._queue) >= self.batch_size:
            await self._process_batch()
        
        return item
    
    async def flush(self):
        """Procesa todos los items pendientes"""
        if self._queue:
            await self._process_batch()
    
    async def _process_batch(self):
        """Procesa un batch de items"""
        if not self._queue or self._processing:
            return
        
        self._processing = True
        batch = []
        
        # Tomar items del queue
        while self._queue and len(batch) < self.batch_size:
            batch.append(self._queue.popleft())
        
        if batch:
            # Procesar batch (esto debe ser implementado por subclases)
            await self._execute_batch(batch)
        
        self._processing = False
    
    async def _execute_batch(self, batch: List[T]):
        """Ejecuta el batch (implementar en subclases)"""
        raise NotImplementedError


class DatabaseBatchProcessor(BatchProcessor):
    """Procesador de batches para base de datos"""
    
    def __init__(self, db_connection, batch_size: int = 100):
        super().__init__(batch_size=batch_size)
        self.db_connection = db_connection
    
    async def _execute_batch(self, batch: List[dict]):
        """Ejecuta batch insert/update"""
        try:
            # Batch insert optimizado
            await self.db_connection.executemany(
                "INSERT INTO ... VALUES ...",
                batch
            )
            logger.debug(f"Processed batch of {len(batch)} items")
        except Exception as e:
            logger.error(f"Batch processing error: {e}")


class CacheBatchProcessor(BatchProcessor):
    """Procesador de batches para cache"""
    
    def __init__(self, cache_client, batch_size: int = 50):
        super().__init__(batch_size=batch_size)
        self.cache_client = cache_client
    
    async def _execute_batch(self, batch: List[tuple]):
        """Ejecuta batch set en cache"""
        try:
            # Pipeline de Redis para batch
            pipe = self.cache_client.pipeline()
            for key, value in batch:
                pipe.set(key, value)
            await pipe.execute()
            logger.debug(f"Cached batch of {len(batch)} items")
        except Exception as e:
            logger.error(f"Cache batch error: {e}")


async def batch_process(
    items: List[T],
    processor: Callable[[List[T]], Awaitable[Any]],
    batch_size: int = 100
) -> List[Any]:
    """
    Procesa items en batches
    
    Args:
        items: Lista de items a procesar
        processor: Función async que procesa un batch
        batch_size: Tamaño del batch
        
    Returns:
        Lista de resultados
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = await processor(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
    
    return results
















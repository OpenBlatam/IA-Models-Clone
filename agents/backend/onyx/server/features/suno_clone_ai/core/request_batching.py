"""
Request Batching
Agrupación de requests para mejor rendimiento
"""

import logging
import asyncio
from typing import List, Any, Callable, Awaitable, TypeVar, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BatchedRequest:
    """Request en un batch"""
    id: str
    data: Any
    future: asyncio.Future
    timestamp: datetime


class RequestBatcher:
    """
    Agrupador de requests
    Agrupa múltiples requests en un solo batch
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_wait: float = 0.05,  # 50ms
        processor: Callable[[List[Any]], Awaitable[List[Any]]] = None
    ):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.processor = processor
        self._queue: List[BatchedRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False
        self._task: Optional[asyncio.Task] = None
    
    async def add(self, data: Any) -> Any:
        """
        Agrega un request al batch
        
        Args:
            data: Datos del request
            
        Returns:
            Resultado del request
        """
        future = asyncio.Future()
        request = BatchedRequest(
            id=str(id(data)),
            data=data,
            future=future,
            timestamp=datetime.utcnow()
        )
        
        async with self._lock:
            self._queue.append(request)
            
            # Procesar si alcanzó el tamaño
            if len(self._queue) >= self.batch_size:
                await self._process_batch()
            elif not self._task:
                # Iniciar task de timeout
                self._task = asyncio.create_task(self._timeout_processor())
        
        return await future
    
    async def _timeout_processor(self):
        """Procesa batch después de timeout"""
        await asyncio.sleep(self.max_wait)
        async with self._lock:
            if self._queue:
                await self._process_batch()
            self._task = None
    
    async def _process_batch(self):
        """Procesa un batch de requests"""
        if self._processing or not self._queue:
            return
        
        self._processing = True
        
        # Tomar requests del queue
        batch = self._queue[:self.batch_size]
        self._queue = self._queue[self.batch_size:]
        
        if self._task:
            self._task.cancel()
            self._task = None
        
        try:
            # Procesar batch
            if self.processor:
                batch_data = [req.data for req in batch]
                results = await self.processor(batch_data)
                
                # Resolver futures
                for i, request in enumerate(batch):
                    if i < len(results):
                        request.future.set_result(results[i])
                    else:
                        request.future.set_exception(ValueError("No result for request"))
            else:
                # Sin processor, resolver con datos originales
                for request in batch:
                    request.future.set_result(request.data)
        
        except Exception as e:
            # Resolver con error
            for request in batch:
                request.future.set_exception(e)
        
        finally:
            self._processing = False
    
    async def flush(self):
        """Procesa todos los requests pendientes"""
        async with self._lock:
            if self._queue:
                await self._process_batch()


# Instancia global
_request_batcher: Optional[RequestBatcher] = None


def get_request_batcher(
    batch_size: int = 10,
    max_wait: float = 0.05,
    processor: Callable[[List[Any]], Awaitable[List[Any]]] = None
) -> RequestBatcher:
    """Obtiene el agrupador de requests"""
    global _request_batcher
    if _request_batcher is None:
        _request_batcher = RequestBatcher(
            batch_size=batch_size,
            max_wait=max_wait,
            processor=processor
        )
    return _request_batcher


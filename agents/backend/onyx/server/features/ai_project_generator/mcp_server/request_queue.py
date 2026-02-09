"""
MCP Request Queue - Cola de requests con prioridad
====================================================
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from heapq import heappush, heappop

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class RequestPriority(int, Enum):
    """Prioridades de request"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class QueuedRequest(BaseModel):
    """Request en la cola"""
    request_id: str = Field(..., description="ID único del request")
    resource_id: str = Field(..., description="ID del recurso")
    operation: str = Field(..., description="Operación")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: RequestPriority = Field(default=RequestPriority.NORMAL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    handler: Optional[Callable] = None
    
    def __lt__(self, other):
        """Comparación para heap (mayor prioridad primero)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class RequestQueue:
    """
    Cola de requests con prioridad
    
    Permite encolar requests y procesarlos según prioridad.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Tamaño máximo de la cola
        """
        self.max_size = max_size
        self._queue: List[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
    
    async def enqueue(
        self,
        request_id: str,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        handler: Optional[Callable] = None,
    ) -> bool:
        """
        Encola un request
        
        Args:
            request_id: ID único del request
            resource_id: ID del recurso
            operation: Operación
            parameters: Parámetros
            priority: Prioridad
            handler: Handler para procesar el request
            
        Returns:
            True si se encoló, False si la cola está llena
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                logger.warning(f"Request queue full, rejecting request {request_id}")
                return False
            
            request = QueuedRequest(
                request_id=request_id,
                resource_id=resource_id,
                operation=operation,
                parameters=parameters,
                priority=priority,
                handler=handler,
            )
            
            heappush(self._queue, request)
            logger.info(f"Enqueued request {request_id} with priority {priority.value}")
            return True
    
    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Desencola el request de mayor prioridad
        
        Returns:
            QueuedRequest o None si la cola está vacía
        """
        async with self._lock:
            if not self._queue:
                return None
            
            return heappop(self._queue)
    
    async def start_processing(self, processor: Callable):
        """
        Inicia procesamiento de la cola
        
        Args:
            processor: Función que procesa requests
        """
        if self._processing:
            return
        
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_loop(processor))
        logger.info("Request queue processing started")
    
    async def stop_processing(self):
        """Detiene procesamiento"""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Request queue processing stopped")
    
    async def _process_loop(self, processor: Callable):
        """Loop de procesamiento"""
        while self._processing:
            try:
                request = await self.dequeue()
                
                if not request:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(request)
                    else:
                        processor(request)
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in request queue processing loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la cola
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "queue_size": len(self._queue),
            "max_size": self.max_size,
            "processing": self._processing,
        }


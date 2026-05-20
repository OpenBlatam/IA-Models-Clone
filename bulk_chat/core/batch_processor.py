"""
Batch Processor - Procesador por Lotes
=======================================

Sistema de procesamiento por lotes con batching inteligente, ventanas de tiempo y procesamiento paralelo.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Estrategia de batching."""
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    HYBRID = "hybrid"


@dataclass
class BatchItem:
    """Item de batch."""
    item_id: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """Batch."""
    batch_id: str
    items: List[BatchItem]
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """Procesador por lotes."""
    
    def __init__(
        self,
        batch_size: int = 100,
        batch_timeout: float = 60.0,
        strategy: BatchStrategy = BatchStrategy.HYBRID,
    ):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.strategy = strategy
        
        self.pending_items: Dict[str, deque] = defaultdict(deque)
        self.batches: Dict[str, Batch] = {}
        self.batch_history: deque = deque(maxlen=100000)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._processing_active = False
    
    def add_item(
        self,
        queue_id: str,
        item_id: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar item a cola."""
        item = BatchItem(
            item_id=item_id,
            data=data,
            metadata=metadata or {},
        )
        
        async def save_item():
            async with self._lock:
                self.pending_items[queue_id].append(item)
        
        asyncio.create_task(save_item())
        
        # Verificar si se debe crear batch
        asyncio.create_task(self._check_and_create_batch(queue_id))
        
        logger.debug(f"Added item {item_id} to queue {queue_id}")
        return item_id
    
    async def _check_and_create_batch(self, queue_id: str):
        """Verificar y crear batch si es necesario."""
        queue = self.pending_items.get(queue_id)
        if not queue:
            return
        
        should_create = False
        
        if self.strategy == BatchStrategy.SIZE_BASED:
            should_create = len(queue) >= self.batch_size
        
        elif self.strategy == BatchStrategy.TIME_BASED:
            if queue:
                oldest_item = queue[0]
                time_since_oldest = (datetime.now() - oldest_item.timestamp).total_seconds()
                should_create = time_since_oldest >= self.batch_timeout
        
        elif self.strategy == BatchStrategy.HYBRID:
            size_check = len(queue) >= self.batch_size
            time_check = False
            if queue:
                oldest_item = queue[0]
                time_since_oldest = (datetime.now() - oldest_item.timestamp).total_seconds()
                time_check = time_since_oldest >= self.batch_timeout
            should_create = size_check or time_check
        
        if should_create:
            await self._create_batch(queue_id)
    
    async def _create_batch(self, queue_id: str):
        """Crear batch desde cola."""
        queue = self.pending_items.get(queue_id)
        if not queue or len(queue) == 0:
            return
        
        # Obtener items para batch
        items_to_batch = []
        batch_size = min(self.batch_size, len(queue))
        
        async with self._lock:
            for _ in range(batch_size):
                if queue:
                    items_to_batch.append(queue.popleft())
        
        if not items_to_batch:
            return
        
        batch_id = f"batch_{queue_id}_{datetime.now().timestamp()}"
        batch = Batch(
            batch_id=batch_id,
            items=items_to_batch,
        )
        
        async with self._lock:
            self.batches[batch_id] = batch
        
        # Procesar batch
        asyncio.create_task(self._process_batch(batch_id, queue_id))
        
        logger.info(f"Created batch {batch_id} with {len(items_to_batch)} items")
    
    async def _process_batch(self, batch_id: str, queue_id: str):
        """Procesar batch."""
        batch = self.batches.get(batch_id)
        if not batch:
            return
        
        batch.status = "processing"
        
        try:
            # En producción, aquí se ejecutaría el procesador real
            # processor = self.processors.get(queue_id)
            # if processor:
            #     await processor(batch.items)
            
            batch.status = "completed"
            batch.processed_at = datetime.now()
        
        except Exception as e:
            batch.status = "failed"
            batch.processed_at = datetime.now()
            logger.error(f"Batch {batch_id} processing failed: {e}")
        
        finally:
            async with self._lock:
                self.batch_history.append(batch)
                if batch_id in self.batches:
                    del self.batches[batch_id]
    
    def register_processor(
        self,
        queue_id: str,
        processor: Callable,
    ):
        """Registrar procesador para cola."""
        # En producción, se guardaría en self.processors
        logger.info(f"Registered processor for queue {queue_id}")
    
    def get_queue_status(self, queue_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estado de cola(s)."""
        if queue_id:
            queue = self.pending_items.get(queue_id, deque())
            return {
                "queue_id": queue_id,
                "pending_items": len(queue),
                "pending_batches": len([b for b in self.batches.values() if b.status == "pending"]),
                "processing_batches": len([b for b in self.batches.values() if b.status == "processing"]),
            }
        else:
            return {
                "total_queues": len(self.pending_items),
                "total_pending_items": sum(len(q) for q in self.pending_items.values()),
                "total_batches": len(self.batches),
            }
    
    def get_batch_history(self, queue_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de batches."""
        history = list(self.batch_history)
        
        if queue_id:
            # Filtrar por queue_id si está en metadata
            history = [b for b in history if queue_id in str(b.metadata)]
        
        history.sort(key=lambda b: b.created_at, reverse=True)
        
        return [
            {
                "batch_id": b.batch_id,
                "item_count": len(b.items),
                "status": b.status,
                "created_at": b.created_at.isoformat(),
                "processed_at": b.processed_at.isoformat() if b.processed_at else None,
            }
            for b in history[:limit]
        ]
    
    def get_batch_processor_summary(self) -> Dict[str, Any]:
        """Obtener resumen del procesador."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for batch in self.batches.values():
            by_status[batch.status] += 1
        
        return {
            "processing_active": self._processing_active,
            "total_queues": len(self.pending_items),
            "total_pending_items": sum(len(q) for q in self.pending_items.values()),
            "total_batches": len(self.batches),
            "batches_by_status": dict(by_status),
            "total_history": len(self.batch_history),
        }



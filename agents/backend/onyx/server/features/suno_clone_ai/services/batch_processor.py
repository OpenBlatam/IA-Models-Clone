"""
Sistema de Procesamiento por Lotes Avanzado

Proporciona:
- Procesamiento en batch
- Priorización de tareas
- Retry automático
- Progreso en tiempo real
- Distribución de carga
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import uuid

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Prioridades de batch"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class BatchStatus(Enum):
    """Estados de batch"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Item de un batch"""
    id: str
    data: Any
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchJob:
    """Trabajo de batch"""
    id: str
    items: List[BatchItem]
    priority: BatchPriority = BatchPriority.NORMAL
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.items:
            raise ValueError("Batch must have at least one item")


class AdvancedBatchProcessor:
    """Procesador de batches avanzado"""
    
    def __init__(
        self,
        max_concurrent_batches: int = 3,
        max_items_per_batch: int = 100,
        worker_pool_size: int = 10
    ):
        """
        Args:
            max_concurrent_batches: Máximo de batches concurrentes
            max_items_per_batch: Máximo de items por batch
            worker_pool_size: Tamaño del pool de workers
        """
        self.max_concurrent_batches = max_concurrent_batches
        self.max_items_per_batch = max_items_per_batch
        self.worker_pool_size = worker_pool_size
        
        self.batches: Dict[str, BatchJob] = {}
        self.processing_queue: deque = deque()
        self.active_batches: set = set()
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        logger.info("AdvancedBatchProcessor initialized")
    
    def create_batch(
        self,
        items: List[Any],
        priority: BatchPriority = BatchPriority.NORMAL,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Crea un nuevo batch
        
        Args:
            items: Lista de items a procesar
            priority: Prioridad del batch
            callback: Callback para progreso
        
        Returns:
            ID del batch
        """
        if len(items) > self.max_items_per_batch:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {self.max_items_per_batch}"
            )
        
        batch_id = str(uuid.uuid4())
        
        batch_items = [
            BatchItem(id=str(uuid.uuid4()), data=item)
            for item in items
        ]
        
        batch = BatchJob(
            id=batch_id,
            items=batch_items,
            priority=priority,
            callback=callback
        )
        
        self.batches[batch_id] = batch
        self.processing_queue.append((priority.value, batch_id))
        
        # Ordenar por prioridad
        self.processing_queue = deque(
            sorted(self.processing_queue, key=lambda x: x[0], reverse=True)
        )
        
        logger.info(f"Batch created: {batch_id} with {len(items)} items")
        return batch_id
    
    async def process_batch(
        self,
        batch_id: str,
        processor_func: Callable[[Any], Any],
        max_retries: int = 3
    ) -> BatchJob:
        """
        Procesa un batch
        
        Args:
            batch_id: ID del batch
            processor_func: Función para procesar cada item
            max_retries: Número máximo de reintentos
        
        Returns:
            BatchJob procesado
        """
        batch = self.batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        
        async with self.semaphore:
            batch.status = BatchStatus.PROCESSING
            batch.started_at = datetime.now()
            self.active_batches.add(batch_id)
            
            try:
                total_items = len(batch.items)
                completed = 0
                
                # Procesar items en paralelo
                tasks = []
                for item in batch.items:
                    task = self._process_item(
                        item,
                        processor_func,
                        max_retries
                    )
                    tasks.append(task)
                
                # Esperar a que todos completen
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Procesar resultados
                for item, result in zip(batch.items, results):
                    if isinstance(result, Exception):
                        item.status = "failed"
                        item.error = str(result)
                    else:
                        item.status = "completed"
                        item.result = result
                        completed += 1
                
                # Actualizar estado del batch
                batch.progress = completed / total_items if total_items > 0 else 0
                
                if completed == total_items:
                    batch.status = BatchStatus.COMPLETED
                elif completed > 0:
                    batch.status = BatchStatus.PARTIAL
                else:
                    batch.status = BatchStatus.FAILED
                
                batch.completed_at = datetime.now()
                
                # Callback si se proporciona
                if batch.callback:
                    try:
                        await batch.callback(batch)
                    except Exception as e:
                        logger.error(f"Error in batch callback: {e}")
                
                logger.info(
                    f"Batch {batch_id} completed: {completed}/{total_items} items"
                )
            
            except Exception as e:
                batch.status = BatchStatus.FAILED
                logger.error(f"Error processing batch {batch_id}: {e}")
            
            finally:
                self.active_batches.discard(batch_id)
        
        return batch
    
    async def _process_item(
        self,
        item: BatchItem,
        processor_func: Callable[[Any], Any],
        max_retries: int
    ) -> Any:
        """Procesa un item individual con retry"""
        for attempt in range(max_retries + 1):
            try:
                # Ejecutar función de procesamiento
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(item.data)
                else:
                    result = await asyncio.to_thread(processor_func, item.data)
                
                item.status = "completed"
                item.result = result
                return result
            
            except Exception as e:
                item.retry_count = attempt + 1
                item.error = str(e)
                
                if attempt < max_retries:
                    # Esperar antes de reintentar (exponential backoff)
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    logger.warning(
                        f"Retrying item {item.id} (attempt {attempt + 1}/{max_retries})"
                    )
                else:
                    item.status = "failed"
                    logger.error(f"Item {item.id} failed after {max_retries} retries: {e}")
                    raise
    
    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Obtiene un batch por ID"""
        return self.batches.get(batch_id)
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancela un batch"""
        batch = self.batches.get(batch_id)
        if batch and batch.status == BatchStatus.PENDING:
            batch.status = BatchStatus.CANCELLED
            return True
        return False
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de batches"""
        total = len(self.batches)
        by_status = {}
        
        for batch in self.batches.values():
            status = batch.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_batches": total,
            "active_batches": len(self.active_batches),
            "queued_batches": len(self.processing_queue),
            "batches_by_status": by_status,
            "max_concurrent": self.max_concurrent_batches
        }


# Instancia global
_batch_processor: Optional[AdvancedBatchProcessor] = None


def get_batch_processor() -> AdvancedBatchProcessor:
    """Obtiene la instancia global del procesador de batches"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = AdvancedBatchProcessor()
    return _batch_processor


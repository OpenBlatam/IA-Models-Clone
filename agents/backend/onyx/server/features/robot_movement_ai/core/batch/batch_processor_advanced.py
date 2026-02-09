"""
Advanced Batch Processor System
================================

Sistema avanzado de procesamiento por lotes.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStatus(Enum):
    """Estado del batch."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchItem:
    """Item del batch."""
    item_id: str
    data: Any
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    processed_at: Optional[str] = None


@dataclass
class Batch:
    """Batch."""
    batch_id: str
    items: List[BatchItem]
    status: BatchStatus = BatchStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedBatchProcessor:
    """
    Procesador avanzado de batches.
    
    Procesa items en lotes con múltiples estrategias.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_workers: int = 5,
        stop_on_error: bool = False
    ):
        """
        Inicializar procesador de batches.
        
        Args:
            batch_size: Tamaño del batch
            max_workers: Número máximo de workers
            stop_on_error: Si detener en error
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stop_on_error = stop_on_error
        self.batches: Dict[str, Batch] = {}
    
    async def process_batch(
        self,
        batch_id: str,
        items: List[Any],
        processor_func: Callable[[Any], R],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Batch:
        """
        Procesar batch de items.
        
        Args:
            batch_id: ID único del batch
            items: Lista de items
            processor_func: Función procesadora
            metadata: Metadata adicional
            
        Returns:
            Batch procesado
        """
        batch_items = [
            BatchItem(
                item_id=f"item_{i}",
                data=item
            )
            for i, item in enumerate(items)
        ]
        
        batch = Batch(
            batch_id=batch_id,
            items=batch_items,
            metadata=metadata or {}
        )
        
        self.batches[batch_id] = batch
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.now().isoformat()
        
        # Procesar items en paralelo
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_item(item: BatchItem) -> None:
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(processor_func):
                        result = await processor_func(item.data)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            processor_func,
                            item.data
                        )
                    
                    item.status = BatchStatus.COMPLETED
                    item.result = result
                    item.processed_at = datetime.now().isoformat()
                except Exception as e:
                    item.status = BatchStatus.FAILED
                    item.error = str(e)
                    item.processed_at = datetime.now().isoformat()
                    
                    if self.stop_on_error:
                        raise
        
        try:
            await asyncio.gather(*[process_item(item) for item in batch_items])
            
            # Determinar estado del batch
            completed = sum(1 for item in batch_items if item.status == BatchStatus.COMPLETED)
            failed = sum(1 for item in batch_items if item.status == BatchStatus.FAILED)
            
            if failed == 0:
                batch.status = BatchStatus.COMPLETED
            elif completed == 0:
                batch.status = BatchStatus.FAILED
            else:
                batch.status = BatchStatus.PARTIAL
            
            batch.completed_at = datetime.now().isoformat()
        except Exception as e:
            batch.status = BatchStatus.FAILED
            batch.completed_at = datetime.now().isoformat()
            logger.error(f"Batch {batch_id} failed: {e}")
        
        return batch
    
    def get_batch(self, batch_id: str) -> Optional[Batch]:
        """Obtener batch por ID."""
        return self.batches.get(batch_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de batches."""
        total_batches = len(self.batches)
        completed = sum(1 for b in self.batches.values() if b.status == BatchStatus.COMPLETED)
        failed = sum(1 for b in self.batches.values() if b.status == BatchStatus.FAILED)
        
        return {
            "total_batches": total_batches,
            "completed_batches": completed,
            "failed_batches": failed,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers
        }


# Instancia global
_advanced_batch_processor: Optional[AdvancedBatchProcessor] = None


def get_advanced_batch_processor(
    batch_size: int = 10,
    max_workers: int = 5
) -> AdvancedBatchProcessor:
    """Obtener instancia global del procesador de batches."""
    global _advanced_batch_processor
    if _advanced_batch_processor is None:
        _advanced_batch_processor = AdvancedBatchProcessor(
            batch_size=batch_size,
            max_workers=max_workers
        )
    return _advanced_batch_processor







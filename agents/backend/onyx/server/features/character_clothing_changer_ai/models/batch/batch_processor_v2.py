"""
Batch Processor V2
=================
Sistema mejorado de procesamiento en lote
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from queue import Queue
import concurrent.futures


class BatchStatus(Enum):
    """Estados de batch"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class BatchItem:
    """Item de batch"""
    id: str
    data: Dict[str, Any]
    status: BatchStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0


@dataclass
class BatchResult:
    """Resultado de batch"""
    batch_id: str
    total_items: int
    completed_items: int
    failed_items: int
    status: BatchStatus
    results: List[Any]
    errors: List[Dict[str, Any]]
    total_time: float
    average_time: float


class BatchProcessorV2:
    """
    Procesador de batch mejorado con paralelismo y retry
    """
    
    def __init__(self, max_workers: int = 4, retry_count: int = 3):
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.batches: Dict[str, Dict] = {}
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    
    def process_batch(
        self,
        batch_id: str,
        items: List[Dict[str, Any]],
        process_func: Callable,
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None
    ) -> BatchResult:
        """
        Procesar batch de items
        
        Args:
            batch_id: ID del batch
            items: Lista de items a procesar
            process_func: Función de procesamiento
            on_progress: Callback de progreso
            on_complete: Callback de completado
        """
        batch_items = [
            BatchItem(
                id=f"{batch_id}_{i}",
                data=item,
                status=BatchStatus.PENDING
            )
            for i, item in enumerate(items)
        ]
        
        self.batches[batch_id] = {
            'items': batch_items,
            'status': BatchStatus.PROCESSING,
            'start_time': time.time(),
            'process_func': process_func
        }
        
        # Procesar con ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.executor = executor
            
            futures = {}
            for item in batch_items:
                future = executor.submit(self._process_item, batch_id, item, process_func)
                futures[future] = item
            
            # Procesar resultados conforme completan
            completed = 0
            failed = 0
            results = []
            errors = []
            
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    item.result = result
                    item.status = BatchStatus.COMPLETED
                    results.append(result)
                    completed += 1
                except Exception as e:
                    item.error = str(e)
                    item.status = BatchStatus.FAILED
                    errors.append({
                        'item_id': item.id,
                        'error': str(e)
                    })
                    failed += 1
                
                # Callback de progreso
                if on_progress:
                    try:
                        on_progress(completed, len(batch_items), failed)
                    except Exception:
                        pass
            
            # Determinar estado final
            if failed == 0:
                final_status = BatchStatus.COMPLETED
            elif completed == 0:
                final_status = BatchStatus.FAILED
            else:
                final_status = BatchStatus.PARTIAL
            
            total_time = time.time() - self.batches[batch_id]['start_time']
            average_time = total_time / len(batch_items) if batch_items else 0
            
            batch_result = BatchResult(
                batch_id=batch_id,
                total_items=len(batch_items),
                completed_items=completed,
                failed_items=failed,
                status=final_status,
                results=results,
                errors=errors,
                total_time=total_time,
                average_time=average_time
            )
            
            self.batches[batch_id]['status'] = final_status
            self.batches[batch_id]['result'] = batch_result
            
            # Callback de completado
            if on_complete:
                try:
                    on_complete(batch_result)
                except Exception:
                    pass
            
            return batch_result
    
    def _process_item(
        self,
        batch_id: str,
        item: BatchItem,
        process_func: Callable
    ) -> Any:
        """Procesar item individual con retry"""
        start_time = time.time()
        
        for attempt in range(self.retry_count):
            try:
                result = process_func(item.data)
                item.processing_time = time.time() - start_time
                item.retry_count = attempt
                return result
            except Exception as e:
                if attempt == self.retry_count - 1:
                    # Último intento falló
                    item.processing_time = time.time() - start_time
                    item.retry_count = attempt + 1
                    raise
                else:
                    # Retry con backoff exponencial
                    time.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado del batch"""
        if batch_id not in self.batches:
            return None
        
        batch = self.batches[batch_id]
        items = batch['items']
        
        completed = len([i for i in items if i.status == BatchStatus.COMPLETED])
        failed = len([i for i in items if i.status == BatchStatus.FAILED])
        processing = len([i for i in items if i.status == BatchStatus.PROCESSING])
        pending = len([i for i in items if i.status == BatchStatus.PENDING])
        
        progress = (completed + failed) / len(items) * 100 if items else 0
        
        return {
            'batch_id': batch_id,
            'status': batch['status'].value,
            'total_items': len(items),
            'completed': completed,
            'failed': failed,
            'processing': processing,
            'pending': pending,
            'progress': progress,
            'elapsed_time': time.time() - batch['start_time'] if 'start_time' in batch else 0
        }
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancelar batch"""
        if batch_id not in self.batches:
            return False
        
        batch = self.batches[batch_id]
        batch['status'] = BatchStatus.CANCELLED
        
        # Cancelar items pendientes
        for item in batch['items']:
            if item.status == BatchStatus.PENDING:
                item.status = BatchStatus.CANCELLED
        
        return True
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de todos los batches"""
        total_batches = len(self.batches)
        total_items = sum(len(b['items']) for b in self.batches.values())
        
        status_counts = {}
        for batch in self.batches.values():
            status = batch['status'].value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_batches': total_batches,
            'total_items': total_items,
            'status_counts': status_counts,
            'max_workers': self.max_workers,
            'retry_count': self.retry_count
        }


# Instancia global
batch_processor_v2 = BatchProcessorV2()


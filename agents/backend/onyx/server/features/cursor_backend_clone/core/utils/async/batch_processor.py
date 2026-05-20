"""
Batch Processor - Procesamiento en lote
=======================================

Sistema para procesar múltiples operaciones en lote de forma eficiente,
con control de concurrencia y manejo de errores robusto.
"""

import asyncio
import logging
from typing import List, TypeVar, Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchResult(Enum):
    """Resultado de procesamiento de un item"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItem:
    """Item individual en un batch"""
    id: str
    data: Any
    result: Optional[Any] = None
    status: BatchResult = BatchResult.SUCCESS
    error: Optional[Exception] = None
    execution_time: float = 0.0


@dataclass
class BatchStats:
    """Estadísticas de procesamiento de batch"""
    total: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float


class BatchProcessor:
    """
    Procesador de operaciones en lote con control de concurrencia.
    
    Permite procesar múltiples items en paralelo con límite de concurrencia,
    manejo de errores individual, y estadísticas detalladas.
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        stop_on_error: bool = False,
        retry_failed: bool = False,
        max_retries: int = 0
    ):
        """
        Inicializar procesador de batch.
        
        Args:
            max_concurrent: Número máximo de operaciones concurrentes
            stop_on_error: Si detener el batch al encontrar un error
            retry_failed: Si reintentar items fallidos
            max_retries: Número máximo de reintentos
        """
        self.max_concurrent = max_concurrent
        self.stop_on_error = stop_on_error
        self.retry_failed = retry_failed
        self.max_retries = max_retries
    
    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        item_id_fn: Optional[Callable[[T], str]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchItem]:
        """
        Procesar lista de items en batch.
        
        Args:
            items: Lista de items a procesar
            processor: Función async que procesa cada item
            item_id_fn: Función para obtener ID de item (default: str(item))
            on_progress: Callback opcional para progreso (current, total)
            
        Returns:
            Lista de BatchItem con resultados
        """
        if not items:
            return []
        
        item_id_fn = item_id_fn or (lambda x: str(x))
        batch_items = [
            BatchItem(id=item_id_fn(item), data=item)
            for item in items
        ]
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        total = len(batch_items)
        completed = 0
        
        async def process_item(item: BatchItem) -> None:
            """Procesar un item individual"""
            nonlocal completed
            
            start_time = datetime.now()
            
            try:
                async with semaphore:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item.data)
                    else:
                        result = processor(item.data)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    item.result = result
                    item.status = BatchResult.SUCCESS
                    item.execution_time = execution_time
                    
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                item.status = BatchResult.FAILED
                item.error = e
                item.execution_time = execution_time
                
                logger.warning(
                    f"⚠️ Batch item {item.id} failed: {type(e).__name__}: {str(e)[:100]}"
                )
                
                if self.stop_on_error:
                    raise
            
            finally:
                completed += 1
                if on_progress:
                    try:
                        if asyncio.iscoroutinefunction(on_progress):
                            await on_progress(completed, total)
                        else:
                            on_progress(completed, total)
                    except Exception as callback_error:
                        logger.error(f"Error in progress callback: {callback_error}")
        
        # Procesar todos los items
        tasks = [process_item(item) for item in batch_items]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Reintentar items fallidos si está habilitado
        if self.retry_failed and self.max_retries > 0:
            failed_items = [item for item in batch_items if item.status == BatchResult.FAILED]
            if failed_items:
                logger.info(f"🔄 Retrying {len(failed_items)} failed items...")
                for retry in range(self.max_retries):
                    retry_items = [item for item in failed_items if item.status == BatchResult.FAILED]
                    if not retry_items:
                        break
                    
                    retry_tasks = [process_item(item) for item in retry_items]
                    await asyncio.gather(*retry_tasks, return_exceptions=True)
        
        return batch_items
    
    def get_stats(self, batch_items: List[BatchItem]) -> BatchStats:
        """
        Calcular estadísticas de procesamiento de batch.
        
        Args:
            batch_items: Lista de BatchItem procesados
            
        Returns:
            BatchStats con estadísticas agregadas
        """
        if not batch_items:
            return BatchStats(
                total=0,
                successful=0,
                failed=0,
                skipped=0,
                total_time=0.0,
                avg_time=0.0,
                min_time=0.0,
                max_time=0.0
            )
        
        successful = sum(1 for item in batch_items if item.status == BatchResult.SUCCESS)
        failed = sum(1 for item in batch_items if item.status == BatchResult.FAILED)
        skipped = sum(1 for item in batch_items if item.status == BatchResult.SKIPPED)
        
        execution_times = [item.execution_time for item in batch_items if item.execution_time > 0]
        
        total_time = sum(execution_times)
        avg_time = total_time / len(execution_times) if execution_times else 0.0
        min_time = min(execution_times) if execution_times else 0.0
        max_time = max(execution_times) if execution_times else 0.0
        
        return BatchStats(
            total=len(batch_items),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time
        )


async def process_batch(
    items: List[T],
    processor: Callable[[T], Any],
    max_concurrent: int = 10,
    stop_on_error: bool = False,
    item_id_fn: Optional[Callable[[T], str]] = None
) -> Tuple[List[BatchItem], BatchStats]:
    """
    Función helper para procesar un batch rápidamente.
    
    Args:
        items: Lista de items a procesar
        processor: Función async que procesa cada item
        max_concurrent: Número máximo de operaciones concurrentes
        stop_on_error: Si detener el batch al encontrar un error
        item_id_fn: Función para obtener ID de item
        
    Returns:
        Tupla de (batch_items, stats)
    """
    batch_processor = BatchProcessor(
        max_concurrent=max_concurrent,
        stop_on_error=stop_on_error
    )
    
    batch_items = await batch_processor.process(
        items=items,
        processor=processor,
        item_id_fn=item_id_fn
    )
    
    stats = batch_processor.get_stats(batch_items)
    
    return batch_items, stats





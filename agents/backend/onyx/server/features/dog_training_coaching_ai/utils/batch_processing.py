"""
Batch Processing Utilities
==========================
Utilidades para procesamiento por lotes.
"""

from typing import List, Any, Callable, Optional, Dict
import asyncio
from datetime import datetime


async def process_batch(
    items: List[Any],
    processor: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5,
    on_progress: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """
    Procesar items en batches con límite de concurrencia.
    
    Args:
        items: Lista de items a procesar
        processor: Función async para procesar cada item
        batch_size: Tamaño de cada batch
        max_concurrent: Número máximo de batches concurrentes
        on_progress: Callback para progreso (procesados, total)
        
    Returns:
        Lista de resultados
    """
    results = []
    total = len(items)
    processed = 0
    
    # Dividir en batches
    batches = [items[i:i + batch_size] for i in range(0, total, batch_size)]
    
    # Procesar batches con límite de concurrencia
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch_items(batch: List[Any]):
        nonlocal processed
        async with semaphore:
            batch_results = await asyncio.gather(*[processor(item) for item in batch])
            processed += len(batch)
            if on_progress:
                on_progress(processed, total)
            return batch_results
    
    batch_results = await asyncio.gather(*[process_batch_items(batch) for batch in batches])
    
    # Aplanar resultados
    for batch_result in batch_results:
        results.extend(batch_result)
    
    return results


async def process_with_retry(
    item: Any,
    processor: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    retry_on: Optional[Callable[[Exception], bool]] = None
) -> Any:
    """
    Procesar item con reintentos.
    
    Args:
        item: Item a procesar
        processor: Función async para procesar
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos
        retry_on: Función para determinar si reintentar
        
    Returns:
        Resultado del procesamiento
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await processor(item)
        except Exception as e:
            last_exception = e
            
            if retry_on and not retry_on(e):
                raise
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
            else:
                raise
    
    raise last_exception


class BatchProcessor:
    """Procesador de batches con estadísticas."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.stats = {
            "total_items": 0,
            "processed": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def process(
        self,
        items: List[Any],
        processor: Callable,
        on_item_complete: Optional[Callable[[Any, Any], None]] = None,
        on_item_error: Optional[Callable[[Any, Exception], None]] = None
    ) -> Dict[str, Any]:
        """
        Procesar items con estadísticas.
        
        Args:
            items: Lista de items
            processor: Función de procesamiento
            on_item_complete: Callback cuando item se completa
            on_item_error: Callback cuando item falla
            
        Returns:
            Diccionario con resultados y estadísticas
        """
        self.stats["total_items"] = len(items)
        self.stats["start_time"] = datetime.now()
        results = []
        errors = []
        
        async def process_with_callbacks(item):
            try:
                result = await processor(item)
                self.stats["processed"] += 1
                if on_item_complete:
                    on_item_complete(item, result)
                return {"success": True, "item": item, "result": result}
            except Exception as e:
                self.stats["failed"] += 1
                error_info = {"item": item, "error": str(e)}
                errors.append(error_info)
                if on_item_error:
                    on_item_error(item, e)
                return {"success": False, **error_info}
        
        batch_results = await process_batch(
            items,
            process_with_callbacks,
            self.batch_size,
            self.max_concurrent
        )
        
        results = [r for r in batch_results if r.get("success")]
        
        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        return {
            "results": results,
            "errors": errors,
            "stats": {
                **self.stats,
                "duration_seconds": duration,
                "success_rate": (self.stats["processed"] / self.stats["total_items"] * 100) if self.stats["total_items"] > 0 else 0
            }
        }


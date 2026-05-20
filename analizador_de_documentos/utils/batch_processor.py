"""
Procesamiento por Lotes Mejorado
=================================

Sistema optimizado para procesar múltiples documentos en paralelo.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Resultado de procesamiento por lotes"""
    total: int
    successful: int
    failed: int
    results: List[Any]
    errors: List[Dict[str, Any]]
    processing_time: float


class BatchProcessor:
    """Procesador de lotes optimizado"""
    
    def __init__(
        self,
        max_workers: int = 10,
        batch_size: int = 100,
        use_processes: bool = False
    ):
        """
        Inicializar procesador de lotes
        
        Args:
            max_workers: Número máximo de workers concurrentes
            batch_size: Tamaño de cada lote
            use_processes: Usar procesos en lugar de threads (para CPU-bound)
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Procesar un lote de items
        
        Args:
            items: Lista de items a procesar
            processor: Función para procesar cada item
            progress_callback: Callback para reportar progreso
        
        Returns:
            BatchResult con resultados
        """
        start_time = time.time()
        results = []
        errors = []
        successful = 0
        failed = 0
        
        # Dividir en batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_item(item):
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        # Ejecutar función síncrona en thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor,
                            processor,
                            item
                        )
                    return {"success": True, "result": result, "item": item}
                except Exception as e:
                    logger.error(f"Error procesando item: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "item": item
                    }
        
        # Procesar todos los items
        processed = 0
        for batch in batches:
            tasks = [process_item(item) for item in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    errors.append({
                        "error": str(result),
                        "item": None
                    })
                    failed += 1
                elif result.get("success"):
                    results.append(result["result"])
                    successful += 1
                else:
                    errors.append({
                        "error": result.get("error", "Unknown error"),
                        "item": result.get("item")
                    })
                    failed += 1
                
                processed += 1
                
                if progress_callback:
                    await progress_callback(processed, len(items), successful, failed)
        
        processing_time = time.time() - start_time
        
        return BatchResult(
            total=len(items),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            processing_time=processing_time
        )
    
    async def process_parallel(
        self,
        items: List[Any],
        processor: Callable,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Procesar items en paralelo sin agrupar en batches
        
        Args:
            items: Lista de items
            processor: Función de procesamiento
            chunk_size: Tamaño de chunk (opcional)
        
        Returns:
            Lista de resultados
        """
        if chunk_size is None:
            chunk_size = self.max_workers
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_item(item):
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        return await processor(item)
                    else:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            self.executor,
                            processor,
                            item
                        )
                except Exception as e:
                    logger.error(f"Error procesando item: {e}")
                    return None
        
        # Procesar en chunks
        results = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_results = await asyncio.gather(
                *[process_item(item) for item in chunk],
                return_exceptions=True
            )
            results.extend([
                r for r in chunk_results
                if r is not None and not isinstance(r, Exception)
            ])
        
        return results
    
    def __del__(self):
        """Limpiar executor"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

















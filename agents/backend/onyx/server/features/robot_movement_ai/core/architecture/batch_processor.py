"""
Procesador de batch para Robot Movement AI v2.0
Procesamiento en lotes con chunking y paralelización
"""

import asyncio
from typing import List, Callable, Any, Optional, TypeVar, Awaitable
from dataclasses import dataclass
from datetime import datetime
import time

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult:
    """Resultado de procesamiento de batch"""
    total_items: int
    processed_items: int
    failed_items: int
    duration: float
    results: List[Any]
    errors: List[Exception]


class BatchProcessor:
    """Procesador de batch con chunking y paralelización"""
    
    def __init__(
        self,
        chunk_size: int = 100,
        max_workers: int = 10,
        stop_on_error: bool = False
    ):
        """
        Inicializar procesador
        
        Args:
            chunk_size: Tamaño de cada chunk
            max_workers: Número máximo de workers concurrentes
            stop_on_error: Detener en caso de error
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.stop_on_error = stop_on_error
    
    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Procesar batch de items
        
        Args:
            items: Lista de items a procesar
            processor: Función async para procesar cada item
            progress_callback: Callback de progreso (current, total)
            
        Returns:
            Resultado del procesamiento
        """
        start_time = time.time()
        results = []
        errors = []
        processed = 0
        failed = 0
        
        # Dividir en chunks
        chunks = [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]
        
        # Procesar chunks con semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_chunk(chunk: List[T]):
            nonlocal processed, failed
            async with semaphore:
                chunk_results = []
                chunk_errors = []
                
                for item in chunk:
                    try:
                        result = await processor(item)
                        chunk_results.append(result)
                        processed += 1
                        
                        if progress_callback:
                            await asyncio.to_thread(progress_callback, processed, len(items))
                    except Exception as e:
                        chunk_errors.append(e)
                        failed += 1
                        
                        if self.stop_on_error:
                            raise
                
                return chunk_results, chunk_errors
        
        # Procesar todos los chunks
        chunk_tasks = [process_chunk(chunk) for chunk in chunks]
        chunk_outputs = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # Consolidar resultados
        for output in chunk_outputs:
            if isinstance(output, Exception):
                errors.append(output)
                failed += 1
            else:
                chunk_results, chunk_errors = output
                results.extend(chunk_results)
                errors.extend(chunk_errors)
        
        duration = time.time() - start_time
        
        return BatchResult(
            total_items=len(items),
            processed_items=processed,
            failed_items=failed,
            duration=duration,
            results=results,
            errors=errors
        )
    
    async def process_sequential(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[R]]
    ) -> BatchResult:
        """Procesar items secuencialmente"""
        start_time = time.time()
        results = []
        errors = []
        processed = 0
        failed = 0
        
        for item in items:
            try:
                result = await processor(item)
                results.append(result)
                processed += 1
            except Exception as e:
                errors.append(e)
                failed += 1
                
                if self.stop_on_error:
                    break
        
        duration = time.time() - start_time
        
        return BatchResult(
            total_items=len(items),
            processed_items=processed,
            failed_items=failed,
            duration=duration,
            results=results,
            errors=errors
        )


def create_batch_processor(
    chunk_size: int = 100,
    max_workers: int = 10
) -> BatchProcessor:
    """Crear instancia de procesador de batch"""
    return BatchProcessor(chunk_size=chunk_size, max_workers=max_workers)





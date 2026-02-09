"""
Parallel Processor
Procesamiento paralelo optimizado
"""

import logging
import asyncio
from typing import List, Any, Callable, Awaitable, TypeVar, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ParallelProcessor:
    """Procesador paralelo"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
    
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """Obtiene thread pool"""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._thread_pool
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """Obtiene process pool"""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._process_pool
    
    async def process_parallel(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[Any]],
        max_concurrent: int = 10
    ) -> List[Any]:
        """
        Procesa items en paralelo con límite de concurrencia
        
        Args:
            items: Lista de items
            processor: Función async de procesamiento
            max_concurrent: Máximo concurrente
            
        Returns:
            Lista de resultados
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(item):
            async with semaphore:
                return await processor(item)
        
        tasks = [bounded_process(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_threaded(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        max_workers: int = None
    ) -> List[Any]:
        """
        Procesa items en threads (para operaciones bloqueantes)
        
        Args:
            items: Lista de items
            processor: Función de procesamiento (sync)
            max_workers: Número de workers
            
        Returns:
            Lista de resultados
        """
        pool = self.get_thread_pool()
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(pool, processor, item)
            for item in items
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_multiprocess(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        max_workers: int = None
    ) -> List[Any]:
        """
        Procesa items en procesos separados (para CPU-bound)
        
        Args:
            items: Lista de items
            processor: Función de procesamiento (debe ser pickleable)
            max_workers: Número de workers
            
        Returns:
            Lista de resultados
        """
        pool = self.get_process_pool()
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(pool, processor, item)
            for item in items
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def close(self):
        """Cierra los pools"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)


# Instancia global
_parallel_processor: Optional[ParallelProcessor] = None


def get_parallel_processor() -> ParallelProcessor:
    """Obtiene el procesador paralelo"""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor()
    return _parallel_processor


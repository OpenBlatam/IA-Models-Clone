"""
I/O Optimizer
Optimizaciones de I/O (archivos, red, base de datos)
"""

import logging
import asyncio
from typing import List, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import aiofiles

logger = logging.getLogger(__name__)


class IOOptimizer:
    """Optimizador de I/O"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._executor: ThreadPoolExecutor = None
        self._semaphore = asyncio.Semaphore(max_workers)
    
    def get_executor(self) -> ThreadPoolExecutor:
        """Obtiene thread pool executor"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    async def read_file_async(self, file_path: str) -> bytes:
        """Lee archivo de forma asíncrona optimizada"""
        async with self._semaphore:
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
    
    async def write_file_async(self, file_path: str, data: bytes):
        """Escribe archivo de forma asíncrona optimizada"""
        async with self._semaphore:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
    
    async def parallel_requests(
        self,
        requests: List[Callable[[], Awaitable[Any]]],
        max_concurrent: int = 10
    ) -> List[Any]:
        """
        Ejecuta requests en paralelo con límite de concurrencia
        
        Args:
            requests: Lista de coroutines a ejecutar
            max_concurrent: Máximo de requests concurrentes
            
        Returns:
            Lista de resultados
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_request(req):
            async with semaphore:
                return await req()
        
        tasks = [bounded_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_database_queries(
        self,
        queries: List[Callable[[], Awaitable[Any]]],
        batch_size: int = 10
    ) -> List[Any]:
        """Ejecuta queries de BD en batches"""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = await asyncio.gather(*[q() for q in batch])
            results.extend(batch_results)
        
        return results
    
    def close(self):
        """Cierra el executor"""
        if self._executor:
            self._executor.shutdown(wait=True)


# Instancia global
_io_optimizer: Optional[IOOptimizer] = None


def get_io_optimizer() -> IOOptimizer:
    """Obtiene el optimizador de I/O"""
    global _io_optimizer
    if _io_optimizer is None:
        _io_optimizer = IOOptimizer()
    return _io_optimizer
















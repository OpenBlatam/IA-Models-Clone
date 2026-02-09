"""
Async Optimizations - Optimizaciones asíncronas
==============================================

Optimizaciones para operaciones asíncronas y concurrencia.
"""

import asyncio
import logging
from typing import List, Callable, Any, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class AsyncBatchProcessor:
    """
    Procesador de batches asíncronos optimizado.
    
    Procesa múltiples items en paralelo con control de concurrencia.
    """
    
    def __init__(self, max_concurrent: int = 10, batch_size: int = 100):
        """
        Args:
            max_concurrent: Máximo de operaciones concurrentes
            batch_size: Tamaño de batch
        """
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Procesa batch de items en paralelo.
        
        Args:
            items: Lista de items a procesar
            processor: Función async que procesa un item
            **kwargs: Argumentos adicionales para processor
        
        Returns:
            Lista de resultados
        """
        async def process_item(item):
            async with self.semaphore:
                try:
                    return await processor(item, **kwargs)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    return None
        
        # Procesar en batches
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[process_item(item) for item in batch],
                return_exceptions=True
            )
            results.extend([r for r in batch_results if not isinstance(r, Exception)])
        
        return results


class AsyncConnectionPool:
    """
    Pool de conexiones asíncronas.
    
    Reutiliza conexiones para mejorar performance.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Args:
            max_size: Tamaño máximo del pool
        """
        self.max_size = max_size
        self._pool: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def acquire(self, factory: Callable):
        """
        Adquiere conexión del pool o crea nueva.
        
        Args:
            factory: Función que crea conexión
        
        Returns:
            Conexión
        """
        async with self._lock:
            if self._pool:
                return self._pool.popleft()
            return await factory()
    
    async def release(self, connection: Any):
        """
        Libera conexión al pool.
        
        Args:
            connection: Conexión a liberar
        """
        async with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append(connection)


class AsyncRateLimiter:
    """Rate limiter asíncrono"""
    
    def __init__(self, rate: int, per: float = 1.0):
        """
        Args:
            rate: Número de requests permitidos
            per: Período de tiempo (segundos)
        """
        self.rate = rate
        self.per = per
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Intenta adquirir permiso.
        
        Returns:
            True si se permite, False si se excede el rate
        """
        async with self.lock:
            now = time.time()
            # Limpiar requests antiguos
            while self.requests and self.requests[0] < now - self.per:
                self.requests.popleft()
            
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True
            return False


def optimize_async_operations():
    """Aplica optimizaciones globales de async"""
    # Configurar event loop para mejor performance
    if hasattr(asyncio, 'set_event_loop_policy'):
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for better async performance")
        except ImportError:
            logger.debug("uvloop not available, using default event loop")


class AsyncTaskQueue:
    """Cola de tareas asíncronas optimizada"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.workers = []
        self.running = False
    
    async def start(self):
        """Inicia workers"""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.max_workers)
        ]
    
    async def stop(self):
        """Detiene workers"""
        self.running = False
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def add_task(self, task: Callable, *args, **kwargs):
        """Agrega tarea a la cola"""
        await self.queue.put((task, args, kwargs))
    
    async def _worker(self):
        """Worker que procesa tareas"""
        while self.running:
            try:
                task, args, kwargs = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                await task(*args, **kwargs)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in worker: {e}")
















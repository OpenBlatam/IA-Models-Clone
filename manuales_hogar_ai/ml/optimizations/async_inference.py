"""
Async Inference
===============

Inferencia asíncrona con colas para mejor throughput.
"""

import logging
import asyncio
from typing import Optional, Callable, Any, Dict
from collections import deque
import time

logger = logging.getLogger(__name__)


class AsyncInferenceQueue:
    """Cola de inferencia asíncrona."""
    
    def __init__(
        self,
        inference_fn: Callable,
        max_queue_size: int = 100,
        batch_size: int = 8,
        max_wait_time: float = 0.1
    ):
        """
        Inicializar cola de inferencia.
        
        Args:
            inference_fn: Función de inferencia
            max_queue_size: Tamaño máximo de cola
            batch_size: Tamaño de batch
            max_wait_time: Tiempo máximo de espera (segundos)
        """
        self.inference_fn = inference_fn
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.pending_items = []
        self.last_batch_time = time.time()
        self.processing = False
        self._logger = logger
        
        # Iniciar worker
        asyncio.create_task(self._worker())
    
    async def submit(self, item: Any) -> Any:
        """
        Enviar item para inferencia.
        
        Args:
            item: Item a procesar
        
        Returns:
            Resultado
        """
        future = asyncio.Future()
        await self.queue.put((item, future))
        return await future
    
    async def _worker(self):
        """Worker que procesa items en batch."""
        while True:
            try:
                # Recopilar items
                items = []
                futures = []
                
                # Obtener primer item
                try:
                    item, future = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.max_wait_time
                    )
                    items.append(item)
                    futures.append(future)
                except asyncio.TimeoutError:
                    continue
                
                # Recopilar más items hasta batch_size o timeout
                start_time = time.time()
                while len(items) < self.batch_size:
                    try:
                        timeout = self.max_wait_time - (time.time() - start_time)
                        if timeout <= 0:
                            break
                        
                        item, future = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=timeout
                        )
                        items.append(item)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                # Procesar batch
                if items:
                    try:
                        results = await asyncio.to_thread(
                            self.inference_fn,
                            items
                        )
                        
                        # Resolver futures
                        for i, future in enumerate(futures):
                            if i < len(results):
                                future.set_result(results[i])
                            else:
                                future.set_exception(ValueError("Resultado no disponible"))
                    
                    except Exception as e:
                        # Resolver con error
                        for future in futures:
                            future.set_exception(e)
            
            except Exception as e:
                self._logger.error(f"Error en worker: {str(e)}")
                await asyncio.sleep(0.1)





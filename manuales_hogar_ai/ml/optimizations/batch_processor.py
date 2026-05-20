"""
Procesador por Lotes
===================

Procesamiento optimizado en batch para mejor rendimiento.
"""

import logging
import torch
import numpy as np
from typing import List, Optional, Any, Callable
from functools import wraps
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Procesador optimizado para batches."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_batch_wait: float = 0.1,
        device: str = "cuda"
    ):
        """
        Inicializar procesador de batches.
        
        Args:
            batch_size: Tamaño de batch
            max_batch_wait: Tiempo máximo de espera (segundos)
            device: Dispositivo
        """
        self.batch_size = batch_size
        self.max_batch_wait = max_batch_wait
        self.device = device
        self._pending_items = []
        self._last_batch_time = time.time()
        self._logger = logger
    
    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Procesar items en batch.
        
        Args:
            items: Lista de items
            process_fn: Función de procesamiento
            **kwargs: Argumentos adicionales
        
        Returns:
            Lista de resultados
        """
        try:
            # Procesar en batches
            results = []
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                batch_results = process_fn(batch, **kwargs)
                results.extend(batch_results)
            
            return results
        
        except Exception as e:
            self._logger.error(f"Error procesando batch: {str(e)}")
            raise
    
    async def process_async_batch(
        self,
        items: List[Any],
        process_fn: Callable,
        **kwargs
    ) -> List[Any]:
        """Procesar items en batch de forma asíncrona."""
        import asyncio
        
        try:
            # Dividir en batches
            batches = [
                items[i:i + self.batch_size]
                for i in range(0, len(items), self.batch_size)
            ]
            
            # Procesar batches en paralelo
            tasks = [
                asyncio.to_thread(process_fn, batch, **kwargs)
                for batch in batches
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            # Aplanar resultados
            results = []
            for batch_result in batch_results:
                results.extend(batch_result)
            
            return results
        
        except Exception as e:
            self._logger.error(f"Error procesando batch async: {str(e)}")
            raise


def batch_process(
    batch_size: int = 32,
    max_wait: float = 0.1
):
    """
    Decorador para procesamiento en batch.
    
    Args:
        batch_size: Tamaño de batch
        max_wait: Tiempo máximo de espera
    """
    def decorator(func: Callable):
        processor = BatchProcessor(batch_size=batch_size, max_batch_wait=max_wait)
        
        @wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            return await processor.process_async_batch(items, func, *args, **kwargs)
        
        return wrapper
    return decorator





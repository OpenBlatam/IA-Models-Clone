"""
Async Optimizer
===============

Optimizador asíncrono para operaciones concurrentes.
"""

import logging
import asyncio
from typing import List, Callable, Any, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AsyncConfig:
    """Configuración asíncrona."""
    max_concurrent: int = 10
    timeout_seconds: float = 30.0
    retry_count: int = 3


class AsyncOptimizer:
    """Optimizador asíncrono."""
    
    def __init__(self, config: Optional[AsyncConfig] = None):
        """
        Inicializar optimizador.
        
        Args:
            config: Configuración
        """
        self.config = config or AsyncConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._logger = logger
    
    async def process_concurrent(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Procesar items concurrentemente.
        
        Args:
            items: Lista de items
            processor: Función procesadora
            batch_size: Tamaño de batch (opcional)
        
        Returns:
            Resultados
        """
        if batch_size:
            # Procesar en batches
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = await self._process_batch(batch, processor)
                results.extend(batch_results)
            return results
        else:
            return await self._process_batch(items, processor)
    
    async def _process_batch(
        self,
        items: List[Any],
        processor: Callable
    ) -> List[Any]:
        """Procesar batch."""
        tasks = []
        
        for item in items:
            task = self._process_single(item, processor)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar excepciones
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.error(f"Error processing item {i}: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single(
        self,
        item: Any,
        processor: Callable
    ) -> Any:
        """Procesar item individual con semáforo."""
        async with self.semaphore:
            try:
                if asyncio.iscoroutinefunction(processor):
                    return await asyncio.wait_for(
                        processor(item),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    # Ejecutar en thread pool para funciones sync
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, processor, item),
                        timeout=self.config.timeout_seconds
                    )
            except asyncio.TimeoutError:
                self._logger.error(f"Timeout processing item")
                raise
            except Exception as e:
                self._logger.error(f"Error processing item: {str(e)}")
                raise





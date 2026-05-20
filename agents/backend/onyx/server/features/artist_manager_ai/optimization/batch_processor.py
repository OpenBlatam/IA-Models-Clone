"""
Batch Processor
===============

Procesador de lotes para optimización de rendimiento.
"""

import logging
from typing import List, Callable, Any, Optional, Dict
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuración de batch."""
    batch_size: int = 32
    max_wait_seconds: float = 1.0
    max_concurrent: int = 10


class BatchProcessor:
    """Procesador de lotes."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Inicializar procesador.
        
        Args:
            config: Configuración
        """
        self.config = config or BatchConfig()
        self.queue: List[Any] = []
        self._processing = False
        self._logger = logger
    
    async def add_item(self, item: Any) -> Any:
        """
        Agregar item al batch.
        
        Args:
            item: Item a procesar
        
        Returns:
            Resultado del procesamiento
        """
        self.queue.append(item)
        
        if len(self.queue) >= self.config.batch_size:
            return await self._process_batch()
        
        return None
    
    async def _process_batch(self) -> List[Any]:
        """Procesar batch actual."""
        if not self.queue:
            return []
        
        batch = self.queue[:self.config.batch_size]
        self.queue = self.queue[self.config.batch_size:]
        
        # Procesar batch
        results = []
        for item in batch:
            # Aquí iría el procesamiento real
            results.append(item)
        
        return results
    
    async def flush(self) -> List[Any]:
        """
        Procesar todos los items pendientes.
        
        Returns:
            Resultados
        """
        if not self.queue:
            return []
        
        batch = self.queue.copy()
        self.queue.clear()
        
        results = []
        for item in batch:
            results.append(item)
        
        return results


class AsyncBatchProcessor:
    """Procesador de lotes asíncrono."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Inicializar procesador asíncrono.
        
        Args:
            config: Configuración
        """
        self.config = config or BatchConfig()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, asyncio.Future] = {}
        self._processing_task: Optional[asyncio.Task] = None
        self._logger = logger
    
    async def process_item(
        self,
        item: Any,
        processor: Callable
    ) -> Any:
        """
        Procesar item en batch.
        
        Args:
            item: Item a procesar
            processor: Función procesadora
        
        Returns:
            Resultado
        """
        import uuid
        
        item_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.results[item_id] = future
        
        await self.queue.put((item_id, item, processor))
        
        # Iniciar procesamiento si no está corriendo
        if not self._processing_task or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_loop())
        
        return await future
    
    async def _process_loop(self):
        """Loop de procesamiento."""
        batch = []
        last_process = asyncio.get_event_loop().time()
        
        while True:
            try:
                # Esperar item con timeout
                timeout = self.config.max_wait_seconds
                try:
                    item_data = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=timeout
                    )
                    batch.append(item_data)
                except asyncio.TimeoutError:
                    pass
                
                # Procesar si batch está lleno o timeout
                current_time = asyncio.get_event_loop().time()
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_process >= self.config.max_wait_seconds)
                )
                
                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_process = current_time
                
                # Si no hay más items y batch está vacío, salir
                if self.queue.empty() and not batch:
                    break
            
            except Exception as e:
                self._logger.error(f"Error in batch processing loop: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[tuple]):
        """Procesar batch."""
        tasks = []
        
        for item_id, item, processor in batch:
            task = asyncio.create_task(self._process_single(item_id, item, processor))
            tasks.append(task)
            
            # Limitar concurrencia
            if len(tasks) >= self.config.max_concurrent:
                await asyncio.gather(*tasks)
                tasks = []
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _process_single(
        self,
        item_id: str,
        item: Any,
        processor: Callable
    ):
        """Procesar item individual."""
        try:
            if asyncio.iscoroutinefunction(processor):
                result = await processor(item)
            else:
                result = processor(item)
            
            if item_id in self.results:
                self.results[item_id].set_result(result)
        except Exception as e:
            if item_id in self.results:
                self.results[item_id].set_exception(e)
        finally:
            if item_id in self.results:
                del self.results[item_id]





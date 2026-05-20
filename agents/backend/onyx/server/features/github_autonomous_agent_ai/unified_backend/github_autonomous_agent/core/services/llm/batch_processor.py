"""
Batch Processor - Procesamiento eficiente de múltiples requests.

Sigue principios de optimización y paralelización.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BatchItem:
    """Item en un batch."""
    id: str
    data: Any
    priority: int = 0  # Mayor = más prioritario
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BatchProcessor:
    """
    Procesador de batches para optimizar requests a LLMs.
    
    Características:
    - Agrupación inteligente de requests
    - Procesamiento paralelo
    - Priorización
    - Rate limiting
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_wait_time: float = 0.5,
        max_parallel: int = 5
    ):
        """
        Inicializar BatchProcessor.
        
        Args:
            batch_size: Tamaño máximo del batch
            max_wait_time: Tiempo máximo de espera antes de procesar batch
            max_parallel: Máximo de batches procesados en paralelo
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_parallel = max_parallel
        self.semaphore = asyncio.Semaphore(max_parallel)
        
        self.batches: Dict[str, List[BatchItem]] = {}
        self.batch_timers: Dict[str, datetime] = {}
        self.processing = False
        
    async def add_to_batch(
        self,
        batch_key: str,
        item_id: str,
        data: Any,
        priority: int = 0
    ) -> None:
        """
        Agregar item a un batch.
        
        Args:
            batch_key: Clave del batch
            item_id: ID del item
            data: Datos del item
            priority: Prioridad del item
        """
        if batch_key not in self.batches:
            self.batches[batch_key] = []
            self.batch_timers[batch_key] = datetime.now()
        
        item = BatchItem(
            id=item_id,
            data=data,
            priority=priority
        )
        
        self.batches[batch_key].append(item)
        
        # Ordenar por prioridad
        self.batches[batch_key].sort(key=lambda x: x.priority, reverse=True)
        
        # Si el batch está lleno, procesar inmediatamente
        if len(self.batches[batch_key]) >= self.batch_size:
            await self._process_batch(batch_key)
    
    async def process_batch(
        self,
        batch_key: str,
        processor_func: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> List[Any]:
        """
        Procesar un batch con una función de procesamiento.
        
        Args:
            batch_key: Clave del batch
            processor_func: Función async que procesa una lista de items
            
        Returns:
            Lista de resultados
        """
        if batch_key not in self.batches or not self.batches[batch_key]:
            return []
        
        items = self.batches[batch_key]
        self.batches[batch_key] = []
        
        async with self.semaphore:
            try:
                data_list = [item.data for item in items]
                results = await processor_func(data_list)
                
                logger.info(f"Batch '{batch_key}' procesado: {len(items)} items")
                return results
            except Exception as e:
                logger.error(f"Error procesando batch '{batch_key}': {e}")
                raise
    
    async def _process_batch(self, batch_key: str) -> None:
        """Procesar batch cuando está lleno."""
        # Esta función se puede sobrescribir o usar con callbacks
        pass
    
    async def process_pending_batches(
        self,
        processor_func: Callable[[str, List[Any]], Awaitable[List[Any]]]
    ) -> None:
        """
        Procesar todos los batches pendientes que han esperado suficiente tiempo.
        
        Args:
            processor_func: Función async que procesa (batch_key, items) -> results
        """
        now = datetime.now()
        batches_to_process = []
        
        for batch_key, items in self.batches.items():
            if not items:
                continue
            
            wait_time = (now - self.batch_timers[batch_key]).total_seconds()
            
            if wait_time >= self.max_wait_time or len(items) >= self.batch_size:
                batches_to_process.append(batch_key)
        
        # Procesar batches en paralelo
        tasks = [
            self._process_single_batch(batch_key, processor_func)
            for batch_key in batches_to_process
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_batch(
        self,
        batch_key: str,
        processor_func: Callable[[str, List[Any]], Awaitable[List[Any]]]
    ) -> None:
        """Procesar un batch individual."""
        if batch_key not in self.batches:
            return
        
        items = self.batches.pop(batch_key)
        self.batch_timers.pop(batch_key, None)
        
        async with self.semaphore:
            try:
                data_list = [item.data for item in items]
                await processor_func(batch_key, data_list)
                logger.debug(f"Batch '{batch_key}' procesado: {len(items)} items")
            except Exception as e:
                logger.error(f"Error procesando batch '{batch_key}': {e}")
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de batches."""
        total_items = sum(len(items) for items in self.batches.values())
        
        return {
            "active_batches": len(self.batches),
            "total_pending_items": total_items,
            "average_batch_size": (
                total_items / len(self.batches)
                if self.batches
                else 0
            ),
            "batches_by_key": {
                key: len(items)
                for key, items in self.batches.items()
            }
        }
    
    def clear_batch(self, batch_key: str) -> None:
        """Limpiar un batch específico."""
        self.batches.pop(batch_key, None)
        self.batch_timers.pop(batch_key, None)
    
    def clear_all(self) -> None:
        """Limpiar todos los batches."""
        self.batches.clear()
        self.batch_timers.clear()




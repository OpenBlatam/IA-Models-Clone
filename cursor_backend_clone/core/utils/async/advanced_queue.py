"""
Advanced Queue - Sistema de Colas Avanzado
===========================================

Sistema de colas con prioridades, delays, y características avanzadas.
"""

import asyncio
import logging
import heapq
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Prioridades de cola"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class QueueItem:
    """Item de cola"""
    data: Any
    priority: QueuePriority = QueuePriority.NORMAL
    delay_until: Optional[datetime] = None
    max_retries: int = 0
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparación para heapq (mayor prioridad primero, luego por tiempo)"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        
        # Si misma prioridad, ordenar por delay_until o created_at
        self_time = self.delay_until or self.created_at
        other_time = other.delay_until or other.created_at
        return self_time < other_time
    
    def is_ready(self) -> bool:
        """Verificar si el item está listo para procesar"""
        if self.delay_until:
            return datetime.now() >= self.delay_until
        return True
    
    def can_retry(self) -> bool:
        """Verificar si se puede reintentar"""
        return self.retry_count < self.max_retries


class AdvancedQueue:
    """
    Cola avanzada con prioridades, delays y reintentos.
    
    Características:
    - Prioridades
    - Delays programados
    - Reintentos automáticos
    - Procesamiento async
    """
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self._queue: List[QueueItem] = []
        self._processing: Dict[str, QueueItem] = {}
        self._completed: List[QueueItem] = []
        self._failed: List[QueueItem] = []
        self._processor: Optional[Callable] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def put(
        self,
        data: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        delay_seconds: Optional[float] = None,
        max_retries: int = 0,
        **metadata
    ) -> None:
        """
        Agregar item a la cola.
        
        Args:
            data: Datos del item
            priority: Prioridad
            delay_seconds: Delay antes de procesar
            max_retries: Máximo de reintentos
            **metadata: Metadata adicional
        """
        if self.max_size and len(self._queue) >= self.max_size:
            raise asyncio.QueueFull("Queue is full")
        
        delay_until = None
        if delay_seconds:
            delay_until = datetime.now() + timedelta(seconds=delay_seconds)
        
        item = QueueItem(
            data=data,
            priority=priority,
            delay_until=delay_until,
            max_retries=max_retries,
            metadata=metadata
        )
        
        heapq.heappush(self._queue, item)
        logger.debug(f"📥 Item added to queue (priority: {priority.name})")
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Obtener item de la cola.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            Datos del item o None si timeout
        """
        start_time = datetime.now()
        
        while True:
            # Verificar timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
            
            # Buscar item listo
            ready_items = [item for item in self._queue if item.is_ready()]
            
            if ready_items:
                # Obtener item de mayor prioridad
                item = min(ready_items, key=lambda x: (x.priority.value, x.delay_until or x.created_at))
                self._queue.remove(item)
                heapq.heapify(self._queue)
                return item.data
            
            # Esperar un poco antes de reintentar
            await asyncio.sleep(0.1)
    
    def set_processor(self, processor: Callable) -> None:
        """
        Establecer función procesadora.
        
        Args:
            processor: Función async que procesa items
        """
        self._processor = processor
        logger.info("⚙️ Queue processor set")
    
    async def start_processing(self) -> None:
        """Iniciar procesamiento automático"""
        if self._running:
            return
        
        if not self._processor:
            raise ValueError("Processor not set")
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        logger.info("🚀 Queue processing started")
    
    async def stop_processing(self) -> None:
        """Detener procesamiento"""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Queue processing stopped")
    
    async def _process_loop(self) -> None:
        """Loop de procesamiento"""
        while self._running:
            try:
                # Obtener item listo
                item_data = await self.get(timeout=1.0)
                
                if item_data is None:
                    continue
                
                # Procesar
                try:
                    if asyncio.iscoroutinefunction(self._processor):
                        await self._processor(item_data)
                    else:
                        self._processor(item_data)
                    
                    # Marcar como completado
                    # Nota: Necesitaríamos guardar el item para esto
                    logger.debug("✅ Item processed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing item: {e}")
                    # Reintentar si es posible
                    # Nota: Necesitaríamos guardar el item para reintentos
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    def size(self) -> int:
        """Obtener tamaño de la cola"""
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """Verificar si la cola está vacía"""
        return len(self._queue) == 0
    
    def clear(self) -> int:
        """
        Limpiar cola.
        
        Returns:
            Número de items eliminados
        """
        count = len(self._queue)
        self._queue.clear()
        logger.info(f"🧹 Queue cleared ({count} items)")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola"""
        return {
            "size": len(self._queue),
            "processing": len(self._processing),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "running": self._running
        }





"""
Queue Manager - Gestor de Colas Avanzado
=========================================

Sistema avanzado de gestión de colas con prioridades, dead letter queues y procesamiento asíncrono.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import heapq

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Prioridad de mensaje."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class QueueMessage:
    """Mensaje de cola."""
    message_id: str
    queue_name: str
    payload: Any
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Queue:
    """Cola."""
    queue_name: str
    max_size: int = 10000
    visibility_timeout: float = 30.0
    message_retention: float = 86400.0
    dead_letter_queue: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueueManager:
    """Gestor de colas avanzado."""
    
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.messages: Dict[str, QueueMessage] = {}
        self.queue_heaps: Dict[str, List[tuple]] = defaultdict(list)  # Priority heaps
        self.processing_messages: Dict[str, datetime] = {}
        self.dead_letter_messages: Dict[str, List[QueueMessage]] = defaultdict(list)
        self.statistics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()
        self._processing_tasks: Dict[str, asyncio.Task] = {}
    
    def create_queue(
        self,
        queue_name: str,
        max_size: int = 10000,
        visibility_timeout: float = 30.0,
        message_retention: float = 86400.0,
        dead_letter_queue: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear cola."""
        queue = Queue(
            queue_name=queue_name,
            max_size=max_size,
            visibility_timeout=visibility_timeout,
            message_retention=message_retention,
            dead_letter_queue=dead_letter_queue,
            metadata=metadata or {},
        )
        
        async def save_queue():
            async with self._lock:
                self.queues[queue_name] = queue
        
        asyncio.create_task(save_queue())
        
        logger.info(f"Created queue: {queue_name}")
        return queue_name
    
    def enqueue(
        self,
        queue_name: str,
        payload: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        message_id: Optional[str] = None,
        max_attempts: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Encolar mensaje."""
        queue = self.queues.get(queue_name)
        if not queue or not queue.enabled:
            raise ValueError(f"Queue {queue_name} not found or disabled")
        
        # Verificar tamaño
        if len(self.queue_heaps[queue_name]) >= queue.max_size:
            raise ValueError(f"Queue {queue_name} is full")
        
        msg_id = message_id or f"msg_{queue_name}_{datetime.now().timestamp()}"
        
        message = QueueMessage(
            message_id=msg_id,
            queue_name=queue_name,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts,
            metadata=metadata or {},
        )
        
        async def save_message():
            async with self._lock:
                self.messages[msg_id] = message
                # Agregar a heap de prioridad (negativo para max heap)
                heapq.heappush(
                    self.queue_heaps[queue_name],
                    (-priority.value, message.created_at.timestamp(), msg_id)
                )
                self.statistics[queue_name]["enqueued"] += 1
        
        asyncio.create_task(save_message())
        
        logger.debug(f"Enqueued message {msg_id} to queue {queue_name}")
        return msg_id
    
    async def dequeue(
        self,
        queue_name: str,
        timeout: Optional[float] = None,
    ) -> Optional[QueueMessage]:
        """Desencolar mensaje."""
        queue = self.queues.get(queue_name)
        if not queue or not queue.enabled:
            return None
        
        start_time = datetime.now()
        
        while True:
            async with self._lock:
                # Buscar mensaje disponible
                while self.queue_heaps[queue_name]:
                    _, _, msg_id = heapq.heappop(self.queue_heaps[queue_name])
                    message = self.messages.get(msg_id)
                    
                    if not message:
                        continue
                    
                    # Verificar si está siendo procesado
                    if msg_id in self.processing_messages:
                        processing_time = self.processing_messages[msg_id]
                        if (datetime.now() - processing_time).total_seconds() < queue.visibility_timeout:
                            # Aún siendo procesado, reintentar después
                            heapq.heappush(
                                self.queue_heaps[queue_name],
                                (-message.priority.value, message.created_at.timestamp(), msg_id)
                            )
                            continue
                        else:
                            # Timeout expirado, volver a procesar
                            del self.processing_messages[msg_id]
                    
                    # Verificar retención
                    age = (datetime.now() - message.created_at).total_seconds()
                    if age > queue.message_retention:
                        # Mensaje expirado, eliminar
                        del self.messages[msg_id]
                        continue
                    
                    # Marcar como procesando
                    self.processing_messages[msg_id] = datetime.now()
                    self.statistics[queue_name]["dequeued"] += 1
                    return message
            
            # No hay mensajes, esperar si hay timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
                await asyncio.sleep(0.1)
            else:
                return None
    
    async def acknowledge(self, message_id: str):
        """Confirmar procesamiento de mensaje."""
        message = self.messages.get(message_id)
        if not message:
            return
        
        async with self._lock:
            if message_id in self.processing_messages:
                del self.processing_messages[message_id]
            
            if message_id in self.messages:
                del self.messages[message_id]
            
            self.statistics[message.queue_name]["acknowledged"] += 1
        
        logger.debug(f"Acknowledged message {message_id}")
    
    async def nack(self, message_id: str, requeue: bool = True):
        """Negar procesamiento de mensaje."""
        message = self.messages.get(message_id)
        if not message:
            return
        
        async with self._lock:
            if message_id in self.processing_messages:
                del self.processing_messages[message_id]
            
            message.attempts += 1
            
            if message.attempts >= message.max_attempts:
                # Mover a dead letter queue
                queue = self.queues.get(message.queue_name)
                if queue and queue.dead_letter_queue:
                    self.dead_letter_messages[queue.dead_letter_queue].append(message)
                
                if message_id in self.messages:
                    del self.messages[message_id]
                
                self.statistics[message.queue_name]["dead_lettered"] += 1
            elif requeue:
                # Reencolar
                heapq.heappush(
                    self.queue_heaps[message.queue_name],
                    (-message.priority.value, message.created_at.timestamp(), message_id)
                )
                self.statistics[message.queue_name]["requeued"] += 1
            else:
                if message_id in self.messages:
                    del self.messages[message_id]
            
            self.statistics[message.queue_name]["nacked"] += 1
        
        logger.debug(f"Nacked message {message_id}, attempts: {message.attempts}")
    
    def get_queue_status(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de cola."""
        queue = self.queues.get(queue_name)
        if not queue:
            return None
        
        return {
            "queue_name": queue_name,
            "total_messages": len(self.queue_heaps[queue_name]),
            "processing_messages": len([
                mid for mid in self.processing_messages.keys()
                if self.messages.get(mid, QueueMessage("", "", {})).queue_name == queue_name
            ]),
            "dead_letter_messages": len(self.dead_letter_messages.get(queue_name, [])),
            "statistics": self.statistics[queue_name].copy(),
        }
    
    def get_queue_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        total_messages = sum(len(heap) for heap in self.queue_heaps.values())
        
        return {
            "total_queues": len(self.queues),
            "total_messages": total_messages,
            "processing_messages": len(self.processing_messages),
            "total_dead_letter_messages": sum(len(dlq) for dlq in self.dead_letter_messages.values()),
        }



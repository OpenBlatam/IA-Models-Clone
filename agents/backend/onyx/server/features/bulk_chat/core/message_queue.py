"""
Message Queue - Sistema de Cola de Mensajes
==========================================

Sistema de cola de mensajes para procesamiento asíncrono.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Prioridad de mensaje."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Message:
    """Mensaje en cola."""
    message_id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageQueue:
    """Cola de mensajes."""
    
    def __init__(self):
        self.queues: Dict[str, List[Message]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.processing: Dict[str, bool] = {}
        self._lock = asyncio.Lock()
    
    async def enqueue(
        self,
        queue_name: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        max_attempts: int = 3,
    ) -> str:
        """
        Agregar mensaje a la cola.
        
        Args:
            queue_name: Nombre de la cola
            payload: Datos del mensaje
            priority: Prioridad
            max_attempts: Intentos máximos
        
        Returns:
            ID del mensaje
        """
        message_id = f"msg_{datetime.now().timestamp()}_{queue_name}"
        
        message = Message(
            message_id=message_id,
            queue_name=queue_name,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts,
        )
        
        async with self._lock:
            if queue_name not in self.queues:
                self.queues[queue_name] = []
            
            # Insertar según prioridad
            self._insert_by_priority(self.queues[queue_name], message)
        
        logger.debug(f"Enqueued message {message_id} to {queue_name}")
        
        # Procesar si hay handler
        if queue_name in self.handlers:
            asyncio.create_task(self._process_queue(queue_name))
        
        return message_id
    
    def _insert_by_priority(self, queue: List[Message], message: Message):
        """Insertar mensaje según prioridad."""
        priority_order = {
            MessagePriority.URGENT: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.MEDIUM: 2,
            MessagePriority.LOW: 3,
        }
        
        message_priority = priority_order[message.priority]
        
        insert_index = len(queue)
        for i, existing_msg in enumerate(queue):
            if priority_order[existing_msg.priority] > message_priority:
                insert_index = i
                break
        
        queue.insert(insert_index, message)
    
    def register_handler(self, queue_name: str, handler: Callable):
        """Registrar handler para procesar mensajes."""
        self.handlers[queue_name] = handler
        logger.info(f"Registered handler for queue: {queue_name}")
        
        # Iniciar procesamiento
        asyncio.create_task(self._process_queue(queue_name))
    
    async def _process_queue(self, queue_name: str):
        """Procesar cola de mensajes."""
        if self.processing.get(queue_name, False):
            return
        
        self.processing[queue_name] = True
        
        try:
            while True:
                async with self._lock:
                    queue = self.queues.get(queue_name, [])
                    if not queue:
                        break
                    
                    message = queue.pop(0)
                
                handler = self.handlers[queue_name]
                
                try:
                    # Procesar mensaje
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                    
                    logger.debug(f"Processed message {message.message_id}")
                    
                except Exception as e:
                    message.attempts += 1
                    logger.error(f"Error processing message {message.message_id}: {e}")
                    
                    if message.attempts < message.max_attempts:
                        # Re-encolar
                        async with self._lock:
                            self._insert_by_priority(self.queues[queue_name], message)
                    else:
                        logger.error(f"Message {message.message_id} failed after {message.max_attempts} attempts")
                
                await asyncio.sleep(0.1)  # Pequeña pausa entre mensajes
                
        finally:
            self.processing[queue_name] = False
    
    def get_queue_size(self, queue_name: str) -> int:
        """Obtener tamaño de cola."""
        return len(self.queues.get(queue_name, []))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de colas."""
        return {
            "queues": {
                name: {
                    "size": len(queue),
                    "processing": self.processing.get(name, False),
                }
                for name, queue in self.queues.items()
            },
            "total_messages": sum(len(queue) for queue in self.queues.values()),
        }

















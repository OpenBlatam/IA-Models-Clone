"""
Message Queue System
====================

Sistema de cola de mensajes.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Prioridad de mensaje."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class Message:
    """Mensaje."""
    message_id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageQueue:
    """
    Cola de mensajes.
    
    Gestiona colas de mensajes con prioridades.
    """
    
    def __init__(self, name: str):
        """
        Inicializar cola de mensajes.
        
        Args:
            name: Nombre de la cola
        """
        self.name = name
        self.messages: List[Message] = []
        self.processing: Dict[str, Message] = {}  # message_id -> message
        self.dead_letter_queue: List[Message] = []
        self.max_size = 10000
    
    async def enqueue(
        self,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        max_attempts: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Agregar mensaje a la cola.
        
        Args:
            payload: Datos del mensaje
            priority: Prioridad
            max_attempts: Intentos máximos
            metadata: Metadata adicional
            
        Returns:
            Mensaje creado
        """
        message_id = f"msg_{len(self.messages) + len(self.processing)}"
        message = Message(
            message_id=message_id,
            queue_name=self.name,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts,
            metadata=metadata or {}
        )
        
        # Insertar según prioridad
        inserted = False
        for i, existing_msg in enumerate(self.messages):
            if message.priority.value > existing_msg.priority.value:
                self.messages.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            self.messages.append(message)
        
        # Limitar tamaño
        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]
        
        logger.debug(f"Enqueued message in {self.name}: {message_id}")
        
        return message
    
    async def dequeue(self, timeout: float = 5.0) -> Optional[Message]:
        """
        Obtener mensaje de la cola.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            Mensaje o None si timeout
        """
        try:
            while not self.messages:
                await asyncio.sleep(0.1)
                if timeout > 0:
                    timeout -= 0.1
                    if timeout <= 0:
                        return None
            
            message = self.messages.pop(0)
            self.processing[message.message_id] = message
            return message
        except Exception as e:
            logger.error(f"Error dequeuing message: {e}")
            return None
    
    async def acknowledge(self, message_id: str) -> bool:
        """
        Confirmar procesamiento de mensaje.
        
        Args:
            message_id: ID del mensaje
            
        Returns:
            True si se confirmó, False si no existe
        """
        if message_id in self.processing:
            del self.processing[message_id]
            return True
        return False
    
    async def nack(self, message_id: str, requeue: bool = True) -> bool:
        """
        Rechazar mensaje.
        
        Args:
            message_id: ID del mensaje
            requeue: Si se debe reencolar
            
        Returns:
            True si se rechazó, False si no existe
        """
        if message_id not in self.processing:
            return False
        
        message = self.processing[message_id]
        message.attempts += 1
        
        if message.attempts >= message.max_attempts:
            # Mover a dead letter queue
            self.dead_letter_queue.append(message)
            del self.processing[message_id]
            logger.warning(f"Message {message_id} moved to dead letter queue")
        elif requeue:
            # Reencolar
            await self.enqueue(
                message.payload,
                priority=message.priority,
                max_attempts=message.max_attempts,
                metadata=message.metadata
            )
            del self.processing[message_id]
        
        return True
    
    def get_size(self) -> int:
        """Obtener tamaño de la cola."""
        return len(self.messages)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola."""
        return {
            "queue_name": self.name,
            "pending_messages": len(self.messages),
            "processing_messages": len(self.processing),
            "dead_letter_messages": len(self.dead_letter_queue),
            "total_messages": len(self.messages) + len(self.processing) + len(self.dead_letter_queue)
        }


class MessageQueueManager:
    """
    Gestor de colas de mensajes.
    
    Gestiona múltiples colas de mensajes.
    """
    
    def __init__(self):
        """Inicializar gestor de colas."""
        self.queues: Dict[str, MessageQueue] = {}
    
    def create_queue(self, name: str) -> MessageQueue:
        """
        Crear cola de mensajes.
        
        Args:
            name: Nombre de la cola
            
        Returns:
            Cola creada
        """
        queue = MessageQueue(name)
        self.queues[name] = queue
        logger.info(f"Created message queue: {name}")
        return queue
    
    def get_queue(self, name: str) -> Optional[MessageQueue]:
        """Obtener cola por nombre."""
        return self.queues.get(name)
    
    def list_queues(self) -> List[str]:
        """Listar nombres de colas."""
        return list(self.queues.keys())


# Instancia global
_message_queue_manager: Optional[MessageQueueManager] = None


def get_message_queue_manager() -> MessageQueueManager:
    """Obtener instancia global del gestor de colas."""
    global _message_queue_manager
    if _message_queue_manager is None:
        _message_queue_manager = MessageQueueManager()
    return _message_queue_manager







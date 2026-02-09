"""
Message Queue - Sistema de colas de mensajes (RabbitMQ/Kafka-like)
===================================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Prioridades de mensaje"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Message:
    """Mensaje en la cola"""
    id: str
    queue: str
    body: Any
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "queue": self.queue,
            "body": self.body,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "metadata": self.metadata
        }


class MessageQueue:
    """Cola de mensajes"""
    
    def __init__(self, name: str):
        self.name = name
        self.messages: List[Message] = []
        self.consumers: List[Callable] = []
        self.processing = False
    
    def enqueue(
        self,
        body: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Agrega un mensaje a la cola"""
        message = Message(
            id=str(uuid.uuid4()),
            queue=self.name,
            body=body,
            priority=priority,
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
        
        logger.info(f"Mensaje {message.id} agregado a cola {self.name}")
        return message
    
    def dequeue(self) -> Optional[Message]:
        """Obtiene el siguiente mensaje de la cola"""
        if not self.messages:
            return None
        
        return self.messages.pop(0)
    
    def peek(self) -> Optional[Message]:
        """Mira el siguiente mensaje sin removerlo"""
        if not self.messages:
            return None
        return self.messages[0]
    
    def size(self) -> int:
        """Tamaño de la cola"""
        return len(self.messages)
    
    def register_consumer(self, consumer: Callable):
        """Registra un consumer"""
        self.consumers.append(consumer)
    
    async def process(self):
        """Procesa mensajes de la cola"""
        if self.processing:
            return
        
        self.processing = True
        
        while self.messages:
            message = self.dequeue()
            if not message:
                break
            
            # Procesar con todos los consumers
            for consumer in self.consumers:
                try:
                    if asyncio.iscoroutinefunction(consumer):
                        await consumer(message)
                    else:
                        consumer(message)
                except Exception as e:
                    logger.error(f"Error procesando mensaje {message.id}: {e}")
                    message.attempts += 1
                    
                    # Reintentar si no excedió max_attempts
                    if message.attempts < message.max_attempts:
                        await asyncio.sleep(2 ** message.attempts)  # Exponential backoff
                        self.messages.append(message)
        
        self.processing = False


class MessageQueueSystem:
    """Sistema de colas de mensajes"""
    
    def __init__(self):
        self.queues: Dict[str, MessageQueue] = {}
        self.exchanges: Dict[str, Dict[str, Any]] = {}
        self.routing_rules: Dict[str, List[str]] = {}  # exchange -> queues
        self.message_history: List[Message] = []
    
    def create_queue(self, name: str) -> MessageQueue:
        """Crea una nueva cola"""
        if name in self.queues:
            return self.queues[name]
        
        queue = MessageQueue(name)
        self.queues[name] = queue
        logger.info(f"Cola {name} creada")
        return queue
    
    def create_exchange(self, name: str, exchange_type: str = "direct"):
        """Crea un exchange"""
        self.exchanges[name] = {
            "name": name,
            "type": exchange_type,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Exchange {name} creado")
    
    def bind_queue_to_exchange(self, queue_name: str, exchange_name: str, routing_key: str = ""):
        """Vincula una cola a un exchange"""
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} no existe")
        
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        key = f"{exchange_name}:{routing_key}"
        if key not in self.routing_rules:
            self.routing_rules[key] = []
        
        if queue_name not in self.routing_rules[key]:
            self.routing_rules[key].append(queue_name)
    
    def publish(
        self,
        exchange_name: str,
        routing_key: str,
        body: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Message]:
        """Publica un mensaje en un exchange"""
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} no existe")
        
        exchange = self.exchanges[exchange_name]
        messages = []
        
        # Encontrar colas destino según routing
        key = f"{exchange_name}:{routing_key}"
        target_queues = self.routing_rules.get(key, [])
        
        # Si no hay routing específico, buscar todas las colas del exchange
        if not target_queues:
            target_queues = [q for q in self.queues.keys() if q.startswith(f"{exchange_name}_")]
        
        # Publicar en cada cola destino
        for queue_name in target_queues:
            if queue_name not in self.queues:
                self.create_queue(queue_name)
            
            queue = self.queues[queue_name]
            message = queue.enqueue(body, priority, metadata)
            messages.append(message)
            self.message_history.append(message)
        
        return messages
    
    def subscribe(
        self,
        queue_name: str,
        consumer: Callable,
        auto_ack: bool = True
    ):
        """Suscribe un consumer a una cola"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        queue = self.queues[queue_name]
        queue.register_consumer(consumer)
    
    async def start_processing(self, queue_name: str):
        """Inicia el procesamiento de una cola"""
        if queue_name not in self.queues:
            raise ValueError(f"Cola {queue_name} no existe")
        
        queue = self.queues[queue_name]
        await queue.process()
    
    def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Obtiene estadísticas de una cola"""
        if queue_name not in self.queues:
            return {"error": "Queue not found"}
        
        queue = self.queues[queue_name]
        return {
            "name": queue.name,
            "size": queue.size(),
            "processing": queue.processing,
            "consumers_count": len(queue.consumers)
        }
    
    def list_queues(self) -> List[str]:
        """Lista todas las colas"""
        return list(self.queues.keys())
    
    def purge_queue(self, queue_name: str) -> bool:
        """Limpia una cola"""
        if queue_name not in self.queues:
            return False
        
        queue = self.queues[queue_name]
        queue.messages.clear()
        return True





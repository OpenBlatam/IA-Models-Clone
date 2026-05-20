"""
MCP Message Queue - Integración con message queues
====================================================
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class MessagePriority(str, Enum):
    """Prioridades de mensaje"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Message(BaseModel):
    """Mensaje en la cola"""
    message_id: str = Field(..., description="ID único del mensaje")
    topic: str = Field(..., description="Tópico del mensaje")
    payload: Dict[str, Any] = Field(..., description="Payload del mensaje")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = Field(default=0, description="Número de reintentos")
    max_retries: int = Field(default=3, description="Máximo de reintentos")


class MessageQueue:
    """
    Cola de mensajes simple
    
    Para producción, integrar con RabbitMQ, Kafka, etc.
    """
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._consumers: Dict[str, List[asyncio.Task]] = {}
        self._running = False
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Publica un mensaje
        
        Args:
            topic: Tópico del mensaje
            payload: Payload del mensaje
            priority: Prioridad del mensaje
            metadata: Metadata adicional
            
        Returns:
            ID del mensaje
        """
        import uuid
        
        message = Message(
            message_id=str(uuid.uuid4()),
            topic=topic,
            payload=payload,
            priority=priority,
            metadata=metadata or {},
        )
        
        # Obtener o crear cola para el tópico
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue()
        
        await self._queues[topic].put(message)
        
        logger.info(f"Published message {message.message_id} to topic {topic}")
        
        return message.message_id
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable,
        consumer_id: Optional[str] = None,
    ):
        """
        Suscribe handler a un tópico
        
        Args:
            topic: Tópico a suscribir
            handler: Función que procesa mensajes
            consumer_id: ID del consumer (opcional)
        """
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue()
        
        consumer_id = consumer_id or f"consumer-{len(self._consumers.get(topic, []))}"
        
        async def consumer_loop():
            queue = self._queues[topic]
            while self._running:
                try:
                    message = await queue.get()
                    
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                        
                        logger.info(f"Message {message.message_id} processed")
                        
                    except Exception as e:
                        message.retry_count += 1
                        
                        if message.retry_count < message.max_retries:
                            logger.warning(
                                f"Message {message.message_id} failed, retrying "
                                f"({message.retry_count}/{message.max_retries})"
                            )
                            await queue.put(message)
                        else:
                            logger.error(
                                f"Message {message.message_id} failed after "
                                f"{message.max_retries} retries: {e}"
                            )
                    
                    queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in consumer {consumer_id}: {e}")
        
        task = asyncio.create_task(consumer_loop())
        
        if topic not in self._consumers:
            self._consumers[topic] = []
        
        self._consumers[topic].append(task)
        
        logger.info(f"Subscribed consumer {consumer_id} to topic {topic}")
    
    async def start(self):
        """Inicia procesamiento de mensajes"""
        self._running = True
        logger.info("Message queue started")
    
    async def stop(self):
        """Detiene procesamiento de mensajes"""
        self._running = False
        
        # Cancelar todos los consumers
        for consumers in self._consumers.values():
            for consumer in consumers:
                consumer.cancel()
        
        await asyncio.gather(*[c for consumers in self._consumers.values() for c in consumers], return_exceptions=True)
        
        logger.info("Message queue stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la cola
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "topics": list(self._queues.keys()),
            "queue_sizes": {topic: queue.qsize() for topic, queue in self._queues.items()},
            "consumers": {topic: len(consumers) for topic, consumers in self._consumers.items()},
            "running": self._running,
        }


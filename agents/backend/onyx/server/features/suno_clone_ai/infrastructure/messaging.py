"""
Message Broker Infrastructure
Abstracción para diferentes brokers de mensajería
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class MessageBrokerType(Enum):
    """Tipos de message brokers soportados"""
    SQS = "sqs"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"
    MEMORY = "memory"


class MessageBroker(ABC):
    """Interfaz abstracta para message broker"""
    
    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publica un mensaje en un topic"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """Se suscribe a un topic"""
        pass
    
    @abstractmethod
    async def consume(self, topic: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Consume un mensaje de un topic"""
        pass


class SQSBroker(MessageBroker):
    """Implementación con AWS SQS"""
    
    def __init__(self, queue_url: str, region: str = "us-east-1"):
        self.queue_url = queue_url
        self.region = region
        self._client = None
    
    async def _get_client(self):
        """Obtiene el cliente SQS"""
        if not self._client:
            from aws.services.sqs_service import SQSService
            self._client = SQSService(
                queue_url=self.queue_url,
                region_name=self.region
            )
        return self._client
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publica un mensaje en SQS"""
        client = await self._get_client()
        await client.send_message({
            "topic": topic,
            "message": message
        })
        return True
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """SQS no tiene suscripciones, usar consume en su lugar"""
        logger.warning("SQS doesn't support subscriptions, use consume instead")
        return False
    
    async def consume(self, topic: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Consume un mensaje de SQS"""
        client = await self._get_client()
        messages = await client.receive_messages(
            max_number=1,
            wait_time_seconds=timeout
        )
        if messages:
            return messages[0]['Body']
        return None


class MemoryBroker(MessageBroker):
    """Implementación en memoria para desarrollo"""
    
    def __init__(self):
        self._queues: Dict[str, list] = {}
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publica un mensaje en memoria"""
        if topic not in self._queues:
            self._queues[topic] = []
        self._queues[topic].append(message)
        return True
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """Suscribe callback (no implementado en memoria)"""
        logger.warning("Memory broker doesn't support subscriptions")
        return False
    
    async def consume(self, topic: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Consume un mensaje de memoria"""
        if topic in self._queues and self._queues[topic]:
            return self._queues[topic].pop(0)
        return None


# Factory function
_message_broker: Optional[MessageBroker] = None


def get_message_broker() -> Optional[MessageBroker]:
    """Obtiene la instancia global del message broker"""
    return _message_broker


def create_message_broker(
    broker_type: MessageBrokerType,
    **kwargs
) -> MessageBroker:
    """Crea un message broker según el tipo"""
    if broker_type == MessageBrokerType.SQS:
        return SQSBroker(
            queue_url=kwargs.get('queue_url'),
            region=kwargs.get('region', 'us-east-1')
        )
    elif broker_type == MessageBrokerType.MEMORY:
        return MemoryBroker()
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}")
















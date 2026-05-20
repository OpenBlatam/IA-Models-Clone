"""
Advanced Message Broker - Message Brokers Avanzados
==================================================

Integración avanzada con message brokers:
- RabbitMQ con patrones avanzados
- Kafka con consumer groups
- Event sourcing
- CQRS pattern
- Dead letter queues
"""

import logging
from typing import Optional, Dict, Any, List, Callable, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import json

from .types import EventName, EventData, EventHandler

logger = logging.getLogger(__name__)


class MessageBrokerType(str, Enum):
    """Tipos de message brokers"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS_STREAM = "redis_stream"
    SQS = "sqs"


class MessageBroker(ABC):
    """Interfaz para message brokers"""
    
    @abstractmethod
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> bool: ...
    
    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Any],
        consumer_group: Optional[str] = None
    ) -> bool: ...
    
    @abstractmethod
    async def create_queue(
        self,
        queue_name: str,
        durable: bool = True,
        **kwargs: Any
    ) -> bool: ...


class RabbitMQBroker(MessageBroker):
    """Broker para RabbitMQ con patrones avanzados"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        vhost: str = "/"
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.vhost = vhost
        self._connection: Optional[Any] = None
        self._channel: Optional[Any] = None
    
    def _get_connection(self) -> Any:
        """Obtiene conexión a RabbitMQ"""
        if self._connection is None:
            try:
                import aio_pika
                import asyncio
                
                # Crear conexión
                url = f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/{self.vhost}"
                self._connection = asyncio.run(aio_pika.connect_robust(url))
            except ImportError:
                logger.error("aio_pika not available. Install with: pip install aio-pika")
                raise
        return self._connection
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> bool:
        """Publica mensaje"""
        try:
            import aio_pika
            
            connection = await self._get_connection()
            channel = await connection.channel()
            
            exchange = await channel.declare_exchange(
                topic,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            message_body = json.dumps(message).encode()
            routing = routing_key or ""
            
            await exchange.publish(
                aio_pika.Message(message_body),
                routing_key=routing
            )
            
            return True
        except Exception as e:
            logger.error(f"RabbitMQ publish error: {e}")
            return False
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Any],
        consumer_group: Optional[str] = None
    ) -> bool:
        """Suscribe a un topic"""
        try:
            import aio_pika
            
            connection = await self._get_connection()
            channel = await connection.channel()
            
            exchange = await channel.declare_exchange(
                topic,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            queue_name = consumer_group or f"{topic}_queue"
            queue = await channel.declare_queue(queue_name, durable=True)
            
            await queue.bind(exchange, routing_key="#")
            
            async def process_message(message: Any) -> None:
                async with message.process():
                    try:
                        body = json.loads(message.body.decode())
                        await handler(body)
                    except Exception as e:
                        logger.error(f"Message processing error: {e}")
                        # Enviar a dead letter queue
                        await self._send_to_dlq(topic, message.body)
            
            await queue.consume(process_message)
            return True
        except Exception as e:
            logger.error(f"RabbitMQ subscribe error: {e}")
            return False
    
    async def _send_to_dlq(self, topic: str, message_body: bytes) -> None:
        """Envía mensaje a dead letter queue"""
        try:
            import aio_pika
            
            connection = await self._get_connection()
            channel = await connection.channel()
            
            dlq_name = f"{topic}_dlq"
            queue = await channel.declare_queue(dlq_name, durable=True)
            
            await channel.default_exchange.publish(
                aio_pika.Message(message_body),
                routing_key=dlq_name
            )
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
    
    async def create_queue(
        self,
        queue_name: str,
        durable: bool = True,
        **kwargs: Any
    ) -> bool:
        """Crea una cola"""
        try:
            import aio_pika
            
            connection = await self._get_connection()
            channel = await connection.channel()
            
            await channel.declare_queue(
                queue_name,
                durable=durable,
                **kwargs
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create queue: {e}")
            return False


class KafkaBroker(MessageBroker):
    """Broker para Kafka con consumer groups"""
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        **kwargs: Any
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.config = kwargs
        self._producer: Optional[Any] = None
        self._consumers: Dict[str, Any] = {}
    
    def _get_producer(self) -> Any:
        """Obtiene producer de Kafka"""
        if self._producer is None:
            try:
                from aiokafka import AIOKafkaProducer
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    **self.config
                )
            except ImportError:
                logger.error("aiokafka not available. Install with: pip install aiokafka")
                raise
        return self._producer
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> bool:
        """Publica mensaje a Kafka"""
        try:
            producer = self._get_producer()
            await producer.start()
            
            message_value = json.dumps(message).encode()
            partition = int(routing_key) if routing_key and routing_key.isdigit() else None
            
            await producer.send_and_wait(
                topic,
                value=message_value,
                partition=partition
            )
            return True
        except Exception as e:
            logger.error(f"Kafka publish error: {e}")
            return False
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Any],
        consumer_group: Optional[str] = None
    ) -> bool:
        """Suscribe a un topic de Kafka"""
        try:
            from aiokafka import AIOKafkaConsumer
            
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=consumer_group or f"{topic}_group",
                **self.config
            )
            
            await consumer.start()
            self._consumers[topic] = consumer
            
            async def process_messages() -> None:
                try:
                    async for msg in consumer:
                        try:
                            message = json.loads(msg.value.decode())
                            await handler(message)
                        except Exception as e:
                            logger.error(f"Message processing error: {e}")
                finally:
                    await consumer.stop()
            
            import asyncio
            asyncio.create_task(process_messages())
            return True
        except Exception as e:
            logger.error(f"Kafka subscribe error: {e}")
            return False
    
    async def create_queue(
        self,
        queue_name: str,
        durable: bool = True,
        **kwargs: Any
    ) -> bool:
        """Crea un topic en Kafka (equivalente a queue)"""
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                **self.config
            )
            
            topic = NewTopic(
                name=queue_name,
                num_partitions=kwargs.get("num_partitions", 1),
                replication_factor=kwargs.get("replication_factor", 1)
            )
            
            admin_client.create_topics([topic])
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka topic: {e}")
            return False


class EventSourcing:
    """Event Sourcing pattern implementation"""
    
    def __init__(self, broker: MessageBroker) -> None:
        self.broker = broker
        self.events: List[Dict[str, Any]] = []
    
    async def append_event(
        self,
        aggregate_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Agrega evento al stream"""
        from datetime import datetime
        event = {
            "aggregate_id": aggregate_id,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.events.append(event)
        
        # Publicar a broker
        return await self.broker.publish(
            f"events_{aggregate_id}",
            event
        )
    
    async def get_events(
        self,
        aggregate_id: str
    ) -> List[Dict[str, Any]]:
        """Obtiene eventos de un aggregate"""
        return [
            e for e in self.events
            if e.get("aggregate_id") == aggregate_id
        ]


def get_message_broker(
    broker_type: MessageBrokerType = MessageBrokerType.RABBITMQ,
    **kwargs: Any
) -> Optional[MessageBroker]:
    """
    Obtiene cliente de message broker.
    
    Args:
        broker_type: Tipo de broker
        **kwargs: Configuración específica
    
    Returns:
        Cliente de message broker
    """
    if broker_type == MessageBrokerType.RABBITMQ:
        return RabbitMQBroker(**kwargs)
    elif broker_type == MessageBrokerType.KAFKA:
        bootstrap_servers = kwargs.get("bootstrap_servers", ["localhost:9092"])
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [bootstrap_servers]
        return KafkaBroker(bootstrap_servers=bootstrap_servers, **kwargs)
    else:
        logger.warning(f"Message broker type {broker_type} not implemented")
        return None


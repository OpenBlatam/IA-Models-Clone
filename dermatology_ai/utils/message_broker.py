"""
Message Broker Integration for Event-Driven Architecture
Supports RabbitMQ and Kafka
"""

import json
import asyncio
from typing import Any, Optional, Callable, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BrokerType(str, Enum):
    """Supported message broker types"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    MEMORY = "memory"  # Fallback for development


@dataclass
class Message:
    """Message structure"""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class MessageBroker:
    """
    Abstract message broker interface.
    Supports RabbitMQ and Kafka with fallback to in-memory queue.
    """
    
    def __init__(self, broker_type: BrokerType = BrokerType.MEMORY):
        self.broker_type = broker_type
        self.connected = False
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def connect(self):
        """Connect to message broker"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from message broker"""
        raise NotImplementedError
    
    async def publish(self, topic: str, message: Message):
        """Publish message to topic"""
        raise NotImplementedError
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        raise NotImplementedError
    
    async def consume(self, topic: str):
        """Consume messages from topic"""
        raise NotImplementedError


class RabbitMQBroker(MessageBroker):
    """RabbitMQ implementation"""
    
    def __init__(self, url: str = "amqp://guest:guest@localhost:5672/"):
        super().__init__(BrokerType.RABBITMQ)
        self.url = url
        self.connection = None
        self.channel = None
    
    async def connect(self):
        """Connect to RabbitMQ"""
        try:
            import aio_pika
            
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            self.connected = True
            logger.info("✅ Connected to RabbitMQ")
        except ImportError:
            logger.warning("aio-pika not installed. Install with: pip install aio-pika")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from RabbitMQ"""
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        self.connected = False
        logger.info("Disconnected from RabbitMQ")
    
    async def publish(self, topic: str, message: Message):
        """Publish message to RabbitMQ exchange"""
        if not self.connected:
            await self.connect()
        
        import aio_pika
        
        exchange = await self.channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
        
        message_body = json.dumps({
            "event_type": message.event_type,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
            "message_id": message.message_id,
            "correlation_id": message.correlation_id,
        }).encode()
        
        await exchange.publish(
            aio_pika.Message(message_body),
            routing_key=message.event_type
        )
        logger.debug(f"Published message to {topic}: {message.event_type}")
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to RabbitMQ topic"""
        if not self.connected:
            await self.connect()
        
        import aio_pika
        
        exchange = await self.channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
        queue = await self.channel.declare_queue(exclusive=True)
        
        await queue.bind(exchange, routing_key="#")
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    data = json.loads(message.body.decode())
                    msg = Message(
                        event_type=data["event_type"],
                        payload=data["payload"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        message_id=data.get("message_id"),
                        correlation_id=data.get("correlation_id"),
                    )
                    await callback(msg)
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        
        await queue.consume(process_message)
        logger.info(f"Subscribed to {topic}")


class KafkaBroker(MessageBroker):
    """Kafka implementation"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        super().__init__(BrokerType.KAFKA)
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
    
    async def connect(self):
        """Connect to Kafka"""
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.producer.start()
            self.connected = True
            logger.info("✅ Connected to Kafka")
        except ImportError:
            logger.warning("aiokafka not installed. Install with: pip install aiokafka")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Kafka"""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        self.connected = False
        logger.info("Disconnected from Kafka")
    
    async def publish(self, topic: str, message: Message):
        """Publish message to Kafka topic"""
        if not self.connected:
            await self.connect()
        
        data = {
            "event_type": message.event_type,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
            "message_id": message.message_id,
            "correlation_id": message.correlation_id,
        }
        
        await self.producer.send_and_wait(topic, data)
        logger.debug(f"Published message to {topic}: {message.event_type}")
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to Kafka topic"""
        if not self.connected:
            await self.connect()
        
        from aiokafka import AIOKafkaConsumer
        
        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode())
        )
        await self.consumer.start()
        
        async def consume_messages():
            async for msg in self.consumer:
                try:
                    data = msg.value
                    message = Message(
                        event_type=data["event_type"],
                        payload=data["payload"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        message_id=data.get("message_id"),
                        correlation_id=data.get("correlation_id"),
                    )
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        
        asyncio.create_task(consume_messages())
        logger.info(f"Subscribed to {topic}")


class MemoryBroker(MessageBroker):
    """In-memory broker for development/testing"""
    
    def __init__(self):
        super().__init__(BrokerType.MEMORY)
        self.queues: Dict[str, asyncio.Queue] = {}
        self._consumers: List[asyncio.Task] = []
    
    async def connect(self):
        """Connect (no-op for memory broker)"""
        self.connected = True
        logger.info("Using in-memory message broker")
    
    async def disconnect(self):
        """Disconnect (stop consumers)"""
        for consumer in self._consumers:
            consumer.cancel()
        await asyncio.gather(*self._consumers, return_exceptions=True)
        self.connected = False
    
    async def publish(self, topic: str, message: Message):
        """Publish to in-memory queue"""
        if topic not in self.queues:
            self.queues[topic] = asyncio.Queue()
        
        await self.queues[topic].put(message)
        logger.debug(f"Published message to {topic}: {message.event_type}")
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to in-memory queue"""
        if topic not in self.queues:
            self.queues[topic] = asyncio.Queue()
        
        async def consume():
            while self.connected:
                try:
                    message = await asyncio.wait_for(
                        self.queues[topic].get(),
                        timeout=1.0
                    )
                    await callback(message)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        
        task = asyncio.create_task(consume())
        self._consumers.append(task)
        logger.info(f"Subscribed to {topic}")


def get_message_broker(
    broker_type: Optional[str] = None,
    **kwargs
) -> MessageBroker:
    """
    Factory function to get message broker instance
    
    Args:
        broker_type: Type of broker (rabbitmq, kafka, memory)
        **kwargs: Broker-specific configuration
        
    Returns:
        MessageBroker instance
    """
    import os
    
    broker_type = broker_type or os.getenv("MESSAGE_BROKER_TYPE", "memory")
    broker_type = BrokerType(broker_type.lower())
    
    if broker_type == BrokerType.RABBITMQ:
        url = kwargs.get("url") or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        return RabbitMQBroker(url=url)
    elif broker_type == BrokerType.KAFKA:
        bootstrap_servers = kwargs.get("bootstrap_servers") or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        return KafkaBroker(bootstrap_servers=bootstrap_servers)
    else:
        return MemoryBroker()


# Event publisher helper
class EventPublisher:
    """Helper class for publishing events"""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
    
    async def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        topic: str = "dermatology-events",
        correlation_id: Optional[str] = None
    ):
        """Publish event to message broker"""
        message = Message(
            event_type=event_type,
            payload=payload,
            timestamp=datetime.utcnow(),
            message_id=f"msg_{datetime.utcnow().timestamp()}",
            correlation_id=correlation_id,
        )
        
        await self.broker.publish(topic, message)
        logger.info(f"Published event: {event_type}")
















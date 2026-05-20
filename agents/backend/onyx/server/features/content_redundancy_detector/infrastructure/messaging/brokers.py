"""
Message Broker Implementations
Support for RabbitMQ, Kafka, Redis Streams
Following event-driven architecture patterns
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported message broker types"""
    REDIS_STREAMS = "redis_streams"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    MEMORY = "memory"  # For testing


class MessageBroker(ABC):
    """Abstract message broker interface"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Publish message to topic"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to topic"""
        pass
    
    @abstractmethod
    async def create_topic(self, topic: str, partitions: int = 1) -> None:
        """Create topic/queue"""
        pass


class RedisStreamBroker(MessageBroker):
    """Redis Streams implementation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client = None
    
    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self._client.ping()
            logger.info("Connected to Redis Streams")
        except ImportError:
            logger.error("redis.asyncio not available. Install with: pip install redis[hiredis]")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._client:
            await self._client.close()
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Publish to Redis Stream"""
        if not self._client:
            await self.connect()
        
        message_data = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in message.items()}
        await self._client.xadd(topic, message_data)
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to Redis Stream"""
        if not self._client:
            await self.connect()
        
        # Create consumer group if provided
        if consumer_group:
            try:
                await self._client.xgroup_create(topic, consumer_group, id="0", mkstream=True)
            except Exception:
                pass  # Group may already exist
        
        # Read messages
        while True:
            if consumer_group:
                messages = await self._client.xreadgroup(
                    consumer_group,
                    "worker-1",
                    {topic: ">"},
                    count=1,
                    block=1000
                )
            else:
                messages = await self._client.xread({topic: "$"}, count=1, block=1000)
            
            for topic_name, stream_messages in messages:
                for msg_id, data in stream_messages:
                    try:
                        parsed = {k: json.loads(v) for k, v in data.items()}
                        await callback(parsed)
                        
                        # Acknowledge if using consumer group
                        if consumer_group:
                            await self._client.xack(topic, consumer_group, msg_id)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
    
    async def create_topic(self, topic: str, partitions: int = 1) -> None:
        """Create stream (Redis Streams doesn't need explicit creation)"""
        pass


class RabbitMQBroker(MessageBroker):
    """RabbitMQ implementation using aio-pika"""
    
    def __init__(self, amqp_url: str = "amqp://guest:guest@localhost:5672/"):
        self.amqp_url = amqp_url
        self._connection = None
        self._channel = None
    
    async def connect(self) -> None:
        """Connect to RabbitMQ"""
        try:
            import aio_pika
            self._connection = await aio_pika.connect_robust(self.amqp_url)
            self._channel = await self._connection.channel()
            logger.info("Connected to RabbitMQ")
        except ImportError:
            logger.error("aio-pika not available. Install with: pip install aio-pika")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ"""
        if self._channel:
            await self._channel.close()
        if self._connection:
            await self._connection.close()
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Publish to RabbitMQ exchange"""
        if not self._channel:
            await self.connect()
        
        exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
        
        routing_key = key or ""
        body = json.dumps(message).encode()
        
        await exchange.publish(
            aio_pika.Message(body),
            routing_key=routing_key
        )
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to RabbitMQ queue"""
        if not self._channel:
            await self.connect()
        
        queue_name = f"{topic}.{consumer_group}" if consumer_group else topic
        
        queue = await self._channel.declare_queue(queue_name, durable=True)
        exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
        
        await queue.bind(exchange, routing_key="#")
        
        async def on_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    data = json.loads(message.body.decode())
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        await queue.consume(on_message)
    
    async def create_topic(self, topic: str, partitions: int = 1) -> None:
        """Declare exchange (RabbitMQ)"""
        if not self._channel:
            await self.connect()
        await self._channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC, durable=True)


class KafkaBroker(MessageBroker):
    """Kafka implementation using aiokafka"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer = None
        self._consumer = None
    
    async def connect(self) -> None:
        """Connect to Kafka"""
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
            await self._producer.start()
            logger.info("Connected to Kafka")
        except ImportError:
            logger.error("aiokafka not available. Install with: pip install aiokafka")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka"""
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Publish to Kafka topic"""
        if not self._producer:
            await self.connect()
        
        value = json.dumps(message).encode()
        key_bytes = key.encode() if key else None
        
        await self._producer.send_and_wait(topic, value=value, key=key_bytes)
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to Kafka topic"""
        if not self._consumer:
            from aiokafka import AIOKafkaConsumer
            group_id = consumer_group or "default-group"
            self._consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id
            )
            await self._consumer.start()
        
        async for msg in self._consumer:
            try:
                data = json.loads(msg.value.decode())
                await callback(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def create_topic(self, topic: str, partitions: int = 1) -> None:
        """Create Kafka topic (requires admin client)"""
        try:
            from aiokafka import AIOKafkaAdminClient
            from aiokafka.admin import NewTopic
            
            admin = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            await admin.start()
            
            topic_list = [NewTopic(name=topic, num_partitions=partitions, replication_factor=1)]
            await admin.create_topics(new_topics=topic_list, validate_only=False)
            
            await admin.close()
        except Exception as e:
            logger.warning(f"Could not create topic {topic}: {e}")


class InMemoryBroker(MessageBroker):
    """In-memory broker for testing"""
    
    def __init__(self):
        self._topics: Dict[str, List[Dict[str, Any]]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def connect(self) -> None:
        """No-op for in-memory"""
        pass
    
    async def disconnect(self) -> None:
        """No-op for in-memory"""
        pass
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Publish to in-memory topic"""
        if topic not in self._topics:
            self._topics[topic] = []
        
        self._topics[topic].append(message)
        
        # Notify subscribers
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to in-memory topic"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
    
    async def create_topic(self, topic: str, partitions: int = 1) -> None:
        """Create in-memory topic"""
        if topic not in self._topics:
            self._topics[topic] = []


def create_broker(broker_type: BrokerType, **kwargs) -> MessageBroker:
    """Factory function to create broker instances"""
    brokers = {
        BrokerType.REDIS_STREAMS: RedisStreamBroker,
        BrokerType.RABBITMQ: RabbitMQBroker,
        BrokerType.KAFKA: KafkaBroker,
        BrokerType.MEMORY: InMemoryBroker,
    }
    
    broker_class = brokers.get(broker_type)
    if not broker_class:
        raise ValueError(f"Unsupported broker type: {broker_type}")
    
    return broker_class(**kwargs)







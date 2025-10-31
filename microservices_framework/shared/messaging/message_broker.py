"""
Advanced Message Broker Implementation
Supports: RabbitMQ, Apache Kafka, Redis Pub/Sub, AWS SQS
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Message broker imports
try:
    import aio_pika
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_SQS_AVAILABLE = True
except ImportError:
    AWS_SQS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MessageBrokerType(Enum):
    """Message broker types"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"
    AWS_SQS = "aws_sqs"
    MEMORY = "memory"  # For testing

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class Message:
    """Message structure"""
    id: str
    topic: str
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class BrokerConfig:
    """Message broker configuration"""
    broker_type: MessageBrokerType
    host: str = "localhost"
    port: int = 5672
    username: Optional[str] = None
    password: Optional[str] = None
    virtual_host: str = "/"
    ssl_enabled: bool = False
    connection_timeout: int = 30
    heartbeat: int = 600
    max_retries: int = 3
    retry_delay: float = 1.0

class MessageBroker(ABC):
    """Abstract message broker interface"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to message broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from message broker"""
        pass
    
    @abstractmethod
    async def publish(self, message: Message) -> bool:
        """Publish a message"""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable) -> str:
        """Subscribe to a topic"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from a topic"""
        pass

class RabbitMQBroker(MessageBroker):
    """RabbitMQ message broker implementation"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.connection = None
        self.channel = None
        self.subscriptions: Dict[str, Any] = {}
    
    async def connect(self) -> bool:
        """Connect to RabbitMQ"""
        try:
            if not RABBITMQ_AVAILABLE:
                raise ImportError("aio_pika is required for RabbitMQ support")
            
            # Build connection URL
            url = f"amqp://{self.config.username or 'guest'}:{self.config.password or 'guest'}@{self.config.host}:{self.config.port}{self.config.virtual_host}"
            
            self.connection = await aio_pika.connect_robust(
                url,
                timeout=self.config.connection_timeout,
                heartbeat=self.config.heartbeat
            )
            
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)
            
            logger.info("Connected to RabbitMQ successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from RabbitMQ"""
        try:
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")
    
    async def publish(self, message: Message) -> bool:
        """Publish message to RabbitMQ"""
        try:
            if not self.channel:
                raise RuntimeError("Not connected to RabbitMQ")
            
            # Declare exchange
            exchange = await self.channel.declare_exchange(
                message.topic,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            # Create message
            body = json.dumps(message.payload).encode()
            headers = message.headers.copy()
            headers.update({
                "message_id": message.id,
                "timestamp": str(message.timestamp),
                "priority": str(message.priority.value),
                "correlation_id": message.correlation_id or "",
                "reply_to": message.reply_to or "",
                "ttl": str(message.ttl) if message.ttl else "",
                "retry_count": str(message.retry_count)
            })
            
            # Publish message
            await exchange.publish(
                aio_pika.Message(
                    body=body,
                    headers=headers,
                    priority=message.priority.value,
                    expiration=str(message.ttl * 1000) if message.ttl else None
                ),
                routing_key=message.topic
            )
            
            logger.debug(f"Published message {message.id} to topic {message.topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message {message.id}: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable) -> str:
        """Subscribe to RabbitMQ topic"""
        try:
            if not self.channel:
                raise RuntimeError("Not connected to RabbitMQ")
            
            # Declare exchange
            exchange = await self.channel.declare_exchange(
                topic,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            # Declare queue
            queue = await self.channel.declare_queue(
                f"{topic}_queue_{uuid.uuid4().hex[:8]}",
                durable=True
            )
            
            # Bind queue to exchange
            await queue.bind(exchange, topic)
            
            # Start consuming
            subscription_id = str(uuid.uuid4())
            
            async def message_handler(message: aio_pika.IncomingMessage):
                async with message.process():
                    try:
                        # Parse message
                        payload = json.loads(message.body.decode())
                        headers = dict(message.headers) if message.headers else {}
                        
                        msg = Message(
                            id=headers.get("message_id", str(uuid.uuid4())),
                            topic=topic,
                            payload=payload,
                            headers=headers,
                            priority=MessagePriority(int(headers.get("priority", 5))),
                            timestamp=float(headers.get("timestamp", time.time())),
                            correlation_id=headers.get("correlation_id"),
                            reply_to=headers.get("reply_to"),
                            ttl=int(headers.get("ttl")) if headers.get("ttl") else None,
                            retry_count=int(headers.get("retry_count", 0))
                        )
                        
                        # Call handler
                        if asyncio.iscoroutinefunction(handler):
                            await handler(msg)
                        else:
                            handler(msg)
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            
            # Start consuming
            await queue.consume(message_handler)
            
            self.subscriptions[subscription_id] = {
                "queue": queue,
                "exchange": exchange,
                "handler": message_handler
            }
            
            logger.info(f"Subscribed to topic {topic} with ID {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from RabbitMQ topic"""
        try:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                await subscription["queue"].cancel()
                del self.subscriptions[subscription_id]
                logger.info(f"Unsubscribed from subscription {subscription_id}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscription_id}: {e}")

class KafkaBroker(MessageBroker):
    """Apache Kafka message broker implementation"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.producer = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.subscriptions: Dict[str, Any] = {}
    
    async def connect(self) -> bool:
        """Connect to Kafka"""
        try:
            if not KAFKA_AVAILABLE:
                raise ImportError("aiokafka is required for Kafka support")
            
            # Create producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=f"{self.config.host}:{self.config.port}",
                value_serializer=lambda v: json.dumps(v).encode(),
                key_serializer=lambda k: k.encode() if k else None,
                retry_backoff_ms=1000,
                max_block_ms=10000
            )
            
            await self.producer.start()
            logger.info("Connected to Kafka successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Kafka"""
        try:
            if self.producer:
                await self.producer.stop()
            
            for consumer in self.consumers.values():
                await consumer.stop()
            
            logger.info("Disconnected from Kafka")
        except Exception as e:
            logger.error(f"Error disconnecting from Kafka: {e}")
    
    async def publish(self, message: Message) -> bool:
        """Publish message to Kafka"""
        try:
            if not self.producer:
                raise RuntimeError("Not connected to Kafka")
            
            # Prepare headers
            headers = [
                ("message_id", message.id.encode()),
                ("timestamp", str(message.timestamp).encode()),
                ("priority", str(message.priority.value).encode()),
                ("correlation_id", (message.correlation_id or "").encode()),
                ("reply_to", (message.reply_to or "").encode()),
                ("ttl", str(message.ttl or "").encode()),
                ("retry_count", str(message.retry_count).encode())
            ]
            
            # Add custom headers
            for key, value in message.headers.items():
                headers.append((key, str(value).encode()))
            
            # Publish message
            await self.producer.send_and_wait(
                message.topic,
                value=message.payload,
                key=message.id.encode(),
                headers=headers
            )
            
            logger.debug(f"Published message {message.id} to topic {message.topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message {message.id}: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable) -> str:
        """Subscribe to Kafka topic"""
        try:
            if not KAFKA_AVAILABLE:
                raise ImportError("aiokafka is required for Kafka support")
            
            # Create consumer
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=f"{self.config.host}:{self.config.port}",
                group_id=f"{topic}_group_{uuid.uuid4().hex[:8]}",
                value_deserializer=lambda m: json.loads(m.decode()),
                key_deserializer=lambda k: k.decode() if k else None,
                auto_offset_reset="latest",
                enable_auto_commit=True
            )
            
            await consumer.start()
            
            subscription_id = str(uuid.uuid4())
            self.consumers[subscription_id] = consumer
            
            # Start consuming in background
            async def consume_messages():
                try:
                    async for msg in consumer:
                        try:
                            # Parse headers
                            headers = {}
                            if msg.headers:
                                for key, value in msg.headers:
                                    headers[key.decode()] = value.decode()
                            
                            message = Message(
                                id=headers.get("message_id", msg.key.decode() if msg.key else str(uuid.uuid4())),
                                topic=msg.topic,
                                payload=msg.value,
                                headers=headers,
                                priority=MessagePriority(int(headers.get("priority", 5))),
                                timestamp=float(headers.get("timestamp", time.time())),
                                correlation_id=headers.get("correlation_id"),
                                reply_to=headers.get("reply_to"),
                                ttl=int(headers.get("ttl")) if headers.get("ttl") else None,
                                retry_count=int(headers.get("retry_count", 0))
                            )
                            
                            # Call handler
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                                
                        except Exception as e:
                            logger.error(f"Error processing Kafka message: {e}")
                            
                except Exception as e:
                    logger.error(f"Kafka consumer error: {e}")
            
            # Start consuming task
            task = asyncio.create_task(consume_messages())
            
            self.subscriptions[subscription_id] = {
                "consumer": consumer,
                "task": task
            }
            
            logger.info(f"Subscribed to Kafka topic {topic} with ID {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to Kafka topic {topic}: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from Kafka topic"""
        try:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                subscription["task"].cancel()
                await subscription["consumer"].stop()
                del self.subscriptions[subscription_id]
                logger.info(f"Unsubscribed from Kafka subscription {subscription_id}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from Kafka {subscription_id}: {e}")

class RedisBroker(MessageBroker):
    """Redis Pub/Sub message broker implementation"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.redis = None
        self.pubsub = None
        self.subscriptions: Dict[str, Any] = {}
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            if not REDIS_AVAILABLE:
                raise ImportError("aioredis is required for Redis support")
            
            self.redis = aioredis.from_url(
                f"redis://{self.config.host}:{self.config.port}",
                password=self.config.password,
                db=0
            )
            
            self.pubsub = self.redis.pubsub()
            await self.redis.ping()
            
            logger.info("Connected to Redis successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.redis:
                await self.redis.close()
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    async def publish(self, message: Message) -> bool:
        """Publish message to Redis"""
        try:
            if not self.redis:
                raise RuntimeError("Not connected to Redis")
            
            # Prepare message data
            message_data = {
                "id": message.id,
                "payload": message.payload,
                "headers": message.headers,
                "priority": message.priority.value,
                "timestamp": message.timestamp,
                "correlation_id": message.correlation_id,
                "reply_to": message.reply_to,
                "ttl": message.ttl,
                "retry_count": message.retry_count
            }
            
            # Publish to Redis
            await self.redis.publish(message.topic, json.dumps(message_data))
            
            logger.debug(f"Published message {message.id} to Redis topic {message.topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message {message.id} to Redis: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable) -> str:
        """Subscribe to Redis topic"""
        try:
            if not self.pubsub:
                raise RuntimeError("Not connected to Redis")
            
            await self.pubsub.subscribe(topic)
            
            subscription_id = str(uuid.uuid4())
            
            async def message_handler():
                try:
                    async for message in self.pubsub.listen():
                        if message["type"] == "message":
                            try:
                                # Parse message data
                                data = json.loads(message["data"])
                                
                                msg = Message(
                                    id=data["id"],
                                    topic=topic,
                                    payload=data["payload"],
                                    headers=data["headers"],
                                    priority=MessagePriority(data["priority"]),
                                    timestamp=data["timestamp"],
                                    correlation_id=data["correlation_id"],
                                    reply_to=data["reply_to"],
                                    ttl=data["ttl"],
                                    retry_count=data["retry_count"]
                                )
                                
                                # Call handler
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(msg)
                                else:
                                    handler(msg)
                                    
                            except Exception as e:
                                logger.error(f"Error processing Redis message: {e}")
                                
                except Exception as e:
                    logger.error(f"Redis subscription error: {e}")
            
            # Start message handler task
            task = asyncio.create_task(message_handler())
            
            self.subscriptions[subscription_id] = {
                "topic": topic,
                "task": task
            }
            
            logger.info(f"Subscribed to Redis topic {topic} with ID {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to Redis topic {topic}: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from Redis topic"""
        try:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                subscription["task"].cancel()
                await self.pubsub.unsubscribe(subscription["topic"])
                del self.subscriptions[subscription_id]
                logger.info(f"Unsubscribed from Redis subscription {subscription_id}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from Redis {subscription_id}: {e}")

class MessageBrokerFactory:
    """Factory for creating message brokers"""
    
    @staticmethod
    def create_broker(config: BrokerConfig) -> MessageBroker:
        """Create message broker based on configuration"""
        
        if config.broker_type == MessageBrokerType.RABBITMQ:
            if not RABBITMQ_AVAILABLE:
                raise ImportError("aio_pika is required for RabbitMQ support")
            return RabbitMQBroker(config)
        
        elif config.broker_type == MessageBrokerType.KAFKA:
            if not KAFKA_AVAILABLE:
                raise ImportError("aiokafka is required for Kafka support")
            return KafkaBroker(config)
        
        elif config.broker_type == MessageBrokerType.REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("aioredis is required for Redis support")
            return RedisBroker(config)
        
        else:
            raise ValueError(f"Unsupported broker type: {config.broker_type}")

class EventBus:
    """
    Event bus for microservices communication
    """
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.subscriptions: Dict[str, str] = {}
    
    async def start(self):
        """Start the event bus"""
        await self.broker.connect()
    
    async def stop(self):
        """Stop the event bus"""
        # Unsubscribe from all topics
        for subscription_id in self.subscriptions.values():
            await self.broker.unsubscribe(subscription_id)
        
        await self.broker.disconnect()
    
    async def publish_event(self, event_type: str, data: Dict[str, Any], **kwargs) -> bool:
        """Publish an event"""
        message = Message(
            id=str(uuid.uuid4()),
            topic=f"events.{event_type}",
            payload=data,
            headers=kwargs.get("headers", {}),
            priority=kwargs.get("priority", MessagePriority.NORMAL),
            correlation_id=kwargs.get("correlation_id"),
            reply_to=kwargs.get("reply_to"),
            ttl=kwargs.get("ttl")
        )
        
        return await self.broker.publish(message)
    
    async def subscribe_to_event(self, event_type: str, handler: Callable) -> str:
        """Subscribe to an event type"""
        topic = f"events.{event_type}"
        
        # Add handler to list
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
        # Subscribe to topic if not already subscribed
        if topic not in self.subscriptions:
            subscription_id = await self.broker.subscribe(topic, self._handle_event)
            self.subscriptions[topic] = subscription_id
        
        return f"{event_type}_{len(self.event_handlers[event_type]) - 1}"
    
    async def _handle_event(self, message: Message):
        """Handle incoming event message"""
        event_type = message.topic.replace("events.", "")
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message.payload, message.headers)
                    else:
                        handler(message.payload, message.headers)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")

# Global event bus instance
event_bus: Optional[EventBus] = None

async def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    global event_bus
    if not event_bus:
        # Create default Redis event bus
        config = BrokerConfig(
            broker_type=MessageBrokerType.REDIS,
            host="localhost",
            port=6379
        )
        broker = MessageBrokerFactory.create_broker(config)
        event_bus = EventBus(broker)
        await event_bus.start()
    return event_bus































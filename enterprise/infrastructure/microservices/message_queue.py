from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid
                import aio_pika
            import aio_pika
            import aio_pika
                import redis.asyncio as redis
from typing import Any, List, Dict, Optional
"""
Message Queue Implementation
===========================

Advanced message queue support for microservices:
- RabbitMQ
- Apache Kafka  
- Redis Streams
- Azure Service Bus
- AWS SQS
"""


logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message data structure."""
    id: str
    topic: str
    payload: Any
    headers: Dict[str, str] = None
    timestamp: datetime = None
    retry_count: int = 0
    
    def __post_init__(self) -> Any:
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
        if not self.headers:
            self.headers = {}


class IMessageQueue(ABC):
    """Abstract interface for message queues."""
    
    @abstractmethod
    async def publish(self, topic: str, message: Any, headers: Dict[str, str] = None) -> bool:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Message], None]) -> str:
        """Subscribe to a topic with a message handler."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if message queue is healthy."""
        pass


class RabbitMQService(IMessageQueue):
    """RabbitMQ message queue implementation."""
    
    def __init__(self, connection_url: str = "amqp://guest:guest@localhost:5672/"):
        
    """__init__ function."""
self.connection_url = connection_url
        self.connection = None
        self.channel = None
        self.subscriptions: Dict[str, Any] = {}
        
    async def _ensure_connection(self) -> Any:
        """Ensure RabbitMQ connection is established."""
        if not self.connection:
            try:
                self.connection = await aio_pika.connect_robust(self.connection_url)
                self.channel = await self.connection.channel()
                logger.info("Connected to RabbitMQ")
            except ImportError:
                logger.error("aio_pika not installed. Install with: pip install aio-pika")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
                raise
    
    async def publish(self, topic: str, message: Any, headers: Dict[str, str] = None) -> bool:
        """Publish message to RabbitMQ."""
        try:
            await self._ensure_connection()
            
            # Declare exchange and queue
            exchange = await self.channel.declare_exchange(
                f"{topic}_exchange", 
                aio_pika.ExchangeType.TOPIC
            )
            
            # Create message
            msg = Message(
                id=str(uuid.uuid4()),
                topic=topic,
                payload=message,
                headers=headers or {}
            )
            
            # Publish message
            await exchange.publish(
                aio_pika.Message(
                    body=json.dumps({
                        "id": msg.id,
                        "payload": msg.payload,
                        "headers": msg.headers,
                        "timestamp": msg.timestamp.isoformat()
                    }).encode(),
                    headers=headers or {}
                ),
                routing_key=topic
            )
            
            logger.debug(f"Published message to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to RabbitMQ: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Message], None]) -> str:
        """Subscribe to RabbitMQ topic."""
        try:
            await self._ensure_connection()
            
            # Declare exchange and queue
            exchange = await self.channel.declare_exchange(
                f"{topic}_exchange",
                aio_pika.ExchangeType.TOPIC
            )
            
            queue = await self.channel.declare_queue(
                f"{topic}_queue",
                durable=True
            )
            
            await queue.bind(exchange, routing_key=topic)
            
            # Create consumer
            async def message_handler(message: aio_pika.IncomingMessage):
                
    """message_handler function."""
async with message.process():
                    try:
                        data = json.loads(message.body.decode())
                        msg = Message(
                            id=data["id"],
                            topic=topic,
                            payload=data["payload"],
                            headers=data.get("headers", {}),
                            timestamp=datetime.fromisoformat(data["timestamp"])
                        )
                        await handler(msg)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            
            subscription_id = str(uuid.uuid4())
            consumer_tag = await queue.consume(message_handler)
            self.subscriptions[subscription_id] = consumer_tag
            
            logger.info(f"Subscribed to topic {topic}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error subscribing to RabbitMQ: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from RabbitMQ topic."""
        try:
            if subscription_id in self.subscriptions:
                consumer_tag = self.subscriptions[subscription_id]
                await self.channel.basic_cancel(consumer_tag)
                del self.subscriptions[subscription_id]
                logger.info(f"Unsubscribed from topic")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unsubscribing from RabbitMQ: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check RabbitMQ health."""
        try:
            await self._ensure_connection()
            return self.connection and not self.connection.is_closed
        except:
            return False


class RedisStreamsService(IMessageQueue):
    """Redis Streams message queue implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis_client = None
        self.subscriptions: Dict[str, asyncio.Task] = {}
        
    async def _ensure_connection(self) -> Any:
        """Ensure Redis connection is established."""
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Connected to Redis Streams")
            except ImportError:
                logger.error("redis not installed. Install with: pip install redis")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def publish(self, topic: str, message: Any, headers: Dict[str, str] = None) -> bool:
        """Publish message to Redis Stream."""
        try:
            await self._ensure_connection()
            
            msg_data = {
                "id": str(uuid.uuid4()),
                "payload": json.dumps(message),
                "headers": json.dumps(headers or {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.xadd(f"stream:{topic}", msg_data)
            logger.debug(f"Published message to stream {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to Redis Streams: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Message], None]) -> str:
        """Subscribe to Redis Stream."""
        try:
            await self._ensure_connection()
            
            subscription_id = str(uuid.uuid4())
            consumer_group = f"{topic}_group"
            consumer_name = f"consumer_{subscription_id}"
            
            # Create consumer group
            try:
                await self.redis_client.xgroup_create(
                    f"stream:{topic}", 
                    consumer_group, 
                    id="0", 
                    mkstream=True
                )
            except:
                # Group might already exist
                pass
            
            # Start consumer task
            task = asyncio.create_task(
                self._consume_stream(topic, consumer_group, consumer_name, handler)
            )
            self.subscriptions[subscription_id] = task
            
            logger.info(f"Subscribed to stream {topic}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error subscribing to Redis Streams: {e}")
            return ""
    
    async def _consume_stream(self, topic: str, consumer_group: str, consumer_name: str, handler: Callable):
        """Consumer loop for Redis Stream."""
        while True:
            try:
                messages = await self.redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {f"stream:{topic}": ">"},
                    count=1,
                    block=1000
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            msg = Message(
                                id=fields[b"id"].decode(),
                                topic=topic,
                                payload=json.loads(fields[b"payload"].decode()),
                                headers=json.loads(fields[b"headers"].decode()),
                                timestamp=datetime.fromisoformat(fields[b"timestamp"].decode())
                            )
                            await handler(msg)
                            
                            # Acknowledge message
                            await self.redis_client.xack(f"stream:{topic}", consumer_group, msg_id)
                            
                        except Exception as e:
                            logger.error(f"Error processing stream message: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stream consumer: {e}")
                await asyncio.sleep(1)
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Redis Stream."""
        try:
            if subscription_id in self.subscriptions:
                task = self.subscriptions[subscription_id]
                task.cancel()
                del self.subscriptions[subscription_id]
                logger.info(f"Unsubscribed from stream")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unsubscribing from Redis Streams: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self._ensure_connection()
            await self.redis_client.ping()
            return True
        except:
            return False


class MessageQueueManager:
    """Manager for multiple message queue backends."""
    
    def __init__(self) -> Any:
        self.queues: Dict[str, IMessageQueue] = {}
        self.primary_queue: Optional[str] = None
        
    def add_queue(self, name: str, queue: IMessageQueue, is_primary: bool = False):
        """Add a message queue backend."""
        self.queues[name] = queue
        if is_primary:
            self.primary_queue = name
        logger.info(f"Added message queue backend: {name}")
    
    async def publish(self, topic: str, message: Any, headers: Dict[str, str] = None) -> Dict[str, bool]:
        """Publish message to all queue backends."""
        results = {}
        
        for name, queue in self.queues.items():
            try:
                result = await queue.publish(topic, message, headers)
                results[name] = result
            except Exception as e:
                logger.error(f"Error publishing to {name}: {e}")
                results[name] = False
        
        return results
    
    async def subscribe(self, topic: str, handler: Callable[[Message], None]) -> Dict[str, str]:
        """Subscribe to topic on primary queue."""
        if self.primary_queue and self.primary_queue in self.queues:
            subscription_id = await self.queues[self.primary_queue].subscribe(topic, handler)
            return {self.primary_queue: subscription_id}
        
        return {}
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all message queue backends."""
        results = {}
        
        for name, queue in self.queues.items():
            try:
                results[name] = await queue.health_check()
            except Exception as e:
                logger.error(f"Error checking health of {name}: {e}")
                results[name] = False
        
        return results 
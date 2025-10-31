"""
Message Queue Integration for Email Sequence System

Provides advanced message queue capabilities including Redis Streams,
RabbitMQ, and Apache Kafka support for scalable email processing.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import hashlib

# Message queue imports
import aioredis
import redis
import aio_pika
from aio_pika import connect_robust, Message, DeliveryMode
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio_mqtt

# Models
from ..models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from ..models.subscriber import Subscriber, SubscriberSegment
from ..models.template import EmailTemplate, TemplateVariable
from ..models.campaign import EmailCampaign, CampaignMetrics

logger = logging.getLogger(__name__)

# Constants
MAX_QUEUE_SIZE = 10000
MAX_RETRIES = 3
RETRY_DELAY = 1.0
BATCH_SIZE = 100
STREAM_MAX_LEN = 1000


class QueueType(Enum):
    """Supported queue types"""
    REDIS_STREAMS = "redis_streams"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    MEMORY = "memory"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QueueConfig:
    """Queue configuration"""
    queue_type: QueueType
    connection_string: str
    max_queue_size: int = MAX_QUEUE_SIZE
    max_retries: int = MAX_RETRIES
    retry_delay: float = RETRY_DELAY
    batch_size: int = BATCH_SIZE
    enable_compression: bool = True
    enable_encryption: bool = False
    enable_metrics: bool = True
    enable_dlq: bool = True  # Dead Letter Queue
    dlq_max_retries: int = 3
    stream_max_len: int = STREAM_MAX_LEN


@dataclass
class QueueMessage:
    """Queue message structure"""
    id: str
    topic: str
    data: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    avg_processing_time: float = 0.0
    queue_size: int = 0
    consumer_lag: float = 0.0
    error_rate: float = 0.0


class MessageQueueManager:
    """Advanced message queue manager with multiple backend support"""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.redis_client = None
        self.rabbitmq_connection = None
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.dead_letter_queue: deque = deque(maxlen=config.max_queue_size)
        
        # Performance tracking
        self.metrics = QueueMetrics()
        self.processing_times: List[float] = []
        self.error_count = 0
        
        # Consumer management
        self.consumers: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info(f"Message Queue Manager initialized for {config.queue_type.value}")
    
    async def initialize(self) -> None:
        """Initialize message queue connections"""
        try:
            if self.config.queue_type == QueueType.REDIS_STREAMS:
                await self._initialize_redis_streams()
            elif self.config.queue_type == QueueType.RABBITMQ:
                await self._initialize_rabbitmq()
            elif self.config.queue_type == QueueType.KAFKA:
                await self._initialize_kafka()
            else:
                await self._initialize_memory_queue()
            
            logger.info("Message queue manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize message queue: {e}")
            raise
    
    async def _initialize_redis_streams(self) -> None:
        """Initialize Redis Streams"""
        self.redis_client = redis.from_url(self.config.connection_string)
        await self.redis_client.ping()
        logger.info("Redis Streams initialized")
    
    async def _initialize_rabbitmq(self) -> None:
        """Initialize RabbitMQ connection"""
        self.rabbitmq_connection = await connect_robust(self.config.connection_string)
        logger.info("RabbitMQ connection established")
    
    async def _initialize_kafka(self) -> None:
        """Initialize Kafka producer and consumer"""
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.config.connection_string,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        await self.kafka_producer.start()
        logger.info("Kafka producer initialized")
    
    async def _initialize_memory_queue(self) -> None:
        """Initialize in-memory queue"""
        self.memory_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        logger.info("In-memory queue initialized")
    
    async def publish_message(
        self,
        topic: str,
        data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Publish message to queue"""
        message_id = str(uuid.uuid4())
        message = QueueMessage(
            id=message_id,
            topic=topic,
            data=data,
            priority=priority,
            metadata=metadata or {}
        )
        
        try:
            if self.config.queue_type == QueueType.REDIS_STREAMS:
                await self._publish_redis_stream(message)
            elif self.config.queue_type == QueueType.RABBITMQ:
                await self._publish_rabbitmq(message)
            elif self.config.queue_type == QueueType.KAFKA:
                await self._publish_kafka(message)
            else:
                await self._publish_memory(message)
            
            self.metrics.messages_sent += 1
            logger.debug(f"Message {message_id} published to {topic}")
            
            return message_id
            
        except Exception as e:
            self.metrics.messages_failed += 1
            logger.error(f"Failed to publish message {message_id}: {e}")
            raise
    
    async def _publish_redis_stream(self, message: QueueMessage) -> None:
        """Publish to Redis Streams"""
        stream_key = f"email_sequence:{message.topic}"
        message_data = {
            "id": message.id,
            "data": json.dumps(message.data),
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "metadata": json.dumps(message.metadata)
        }
        
        await self.redis_client.xadd(
            stream_key,
            message_data,
            maxlen=self.config.stream_max_len
        )
    
    async def _publish_rabbitmq(self, message: QueueMessage) -> None:
        """Publish to RabbitMQ"""
        channel = await self.rabbitmq_connection.channel()
        
        # Declare queue
        queue = await channel.declare_queue(
            message.topic,
            durable=True
        )
        
        # Create message
        message_body = json.dumps({
            "id": message.id,
            "data": message.data,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata
        })
        
        # Publish message
        await channel.default_exchange.publish(
            Message(
                body=message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=message.priority.value
            ),
            routing_key=message.topic
        )
    
    async def _publish_kafka(self, message: QueueMessage) -> None:
        """Publish to Kafka"""
        message_data = {
            "id": message.id,
            "data": message.data,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata
        }
        
        await self.kafka_producer.send_and_wait(
            topic=message.topic,
            value=message_data,
            key=message.id.encode()
        )
    
    async def _publish_memory(self, message: QueueMessage) -> None:
        """Publish to memory queue"""
        await self.memory_queue.put(message)
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[QueueMessage], None],
        consumer_group: str = None
    ) -> None:
        """Subscribe to topic with handler"""
        self.message_handlers[topic].append(handler)
        
        if not self.is_running:
            await self._start_consumers()
    
    async def _start_consumers(self) -> None:
        """Start message consumers"""
        self.is_running = True
        
        if self.config.queue_type == QueueType.REDIS_STREAMS:
            await self._start_redis_consumers()
        elif self.config.queue_type == QueueType.RABBITMQ:
            await self._start_rabbitmq_consumers()
        elif self.config.queue_type == QueueType.KAFKA:
            await self._start_kafka_consumers()
        else:
            await self._start_memory_consumers()
    
    async def _start_redis_consumers(self) -> None:
        """Start Redis Streams consumers"""
        for topic in self.message_handlers.keys():
            consumer_task = asyncio.create_task(
                self._consume_redis_stream(topic)
            )
            self.consumers.append(consumer_task)
    
    async def _consume_redis_stream(self, topic: str) -> None:
        """Consume from Redis Streams"""
        stream_key = f"email_sequence:{topic}"
        last_id = "0"
        
        while self.is_running:
            try:
                # Read from stream
                messages = await self.redis_client.xread(
                    {stream_key: last_id},
                    count=self.config.batch_size,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        last_id = message_id
                        
                        # Parse message
                        message = QueueMessage(
                            id=fields[b'id'].decode(),
                            topic=topic,
                            data=json.loads(fields[b'data'].decode()),
                            priority=MessagePriority(fields[b'priority']),
                            timestamp=datetime.fromisoformat(fields[b'timestamp'].decode()),
                            metadata=json.loads(fields[b'metadata'].decode())
                        )
                        
                        # Process message
                        await self._process_message(message)
                
            except Exception as e:
                logger.error(f"Redis stream consumption error: {e}")
                await asyncio.sleep(1)
    
    async def _start_rabbitmq_consumers(self) -> None:
        """Start RabbitMQ consumers"""
        for topic in self.message_handlers.keys():
            consumer_task = asyncio.create_task(
                self._consume_rabbitmq(topic)
            )
            self.consumers.append(consumer_task)
    
    async def _consume_rabbitmq(self, topic: str) -> None:
        """Consume from RabbitMQ"""
        channel = await self.rabbitmq_connection.channel()
        
        # Declare queue
        queue = await channel.declare_queue(topic, durable=True)
        
        async def message_handler(message):
            async with message.process():
                try:
                    # Parse message
                    message_data = json.loads(message.body.decode())
                    queue_message = QueueMessage(
                        id=message_data["id"],
                        topic=topic,
                        data=message_data["data"],
                        priority=MessagePriority(message_data["priority"]),
                        timestamp=datetime.fromisoformat(message_data["timestamp"]),
                        metadata=message_data["metadata"]
                    )
                    
                    # Process message
                    await self._process_message(queue_message)
                    
                except Exception as e:
                    logger.error(f"RabbitMQ message processing error: {e}")
                    # Reject message
                    await message.reject(requeue=False)
        
        await queue.consume(message_handler)
    
    async def _start_kafka_consumers(self) -> None:
        """Start Kafka consumers"""
        for topic in self.message_handlers.keys():
            consumer_task = asyncio.create_task(
                self._consume_kafka(topic)
            )
            self.consumers.append(consumer_task)
    
    async def _consume_kafka(self, topic: str) -> None:
        """Consume from Kafka"""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.config.connection_string,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f"email_sequence_consumer_{topic}"
        )
        
        await consumer.start()
        
        try:
            async for message in consumer:
                try:
                    # Parse message
                    message_data = message.value
                    queue_message = QueueMessage(
                        id=message_data["id"],
                        topic=topic,
                        data=message_data["data"],
                        priority=MessagePriority(message_data["priority"]),
                        timestamp=datetime.fromisoformat(message_data["timestamp"]),
                        metadata=message_data["metadata"]
                    )
                    
                    # Process message
                    await self._process_message(queue_message)
                    
                except Exception as e:
                    logger.error(f"Kafka message processing error: {e}")
                    
        finally:
            await consumer.stop()
    
    async def _start_memory_consumers(self) -> None:
        """Start memory queue consumers"""
        consumer_task = asyncio.create_task(self._consume_memory())
        self.consumers.append(consumer_task)
    
    async def _consume_memory(self) -> None:
        """Consume from memory queue"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.memory_queue.get(),
                    timeout=1.0
                )
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Memory queue consumption error: {e}")
    
    async def _process_message(self, message: QueueMessage) -> None:
        """Process message with handlers"""
        start_time = time.time()
        
        try:
            # Get handlers for topic
            handlers = self.message_handlers.get(message.topic, [])
            
            if not handlers:
                logger.warning(f"No handlers found for topic: {message.topic}")
                return
            
            # Execute handlers
            for handler in handlers:
                try:
                    await asyncio.create_task(
                        self._execute_handler(handler, message)
                    )
                except Exception as e:
                    logger.error(f"Handler execution error: {e}")
                    await self._handle_message_failure(message, e)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.metrics.messages_received += 1
            
            # Update average processing time
            if self.processing_times:
                self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Message processing error: {e}")
            await self._handle_message_failure(message, e)
    
    async def _execute_handler(
        self,
        handler: Callable[[QueueMessage], None],
        message: QueueMessage
    ) -> None:
        """Execute message handler"""
        if asyncio.iscoroutinefunction(handler):
            await handler(message)
        else:
            handler(message)
    
    async def _handle_message_failure(
        self,
        message: QueueMessage,
        error: Exception
    ) -> None:
        """Handle message processing failure"""
        self.metrics.messages_failed += 1
        
        # Increment retry count
        message.retry_count += 1
        
        if message.retry_count < self.config.max_retries:
            # Retry message
            self.metrics.messages_retried += 1
            await asyncio.sleep(self.config.retry_delay * message.retry_count)
            await self.publish_message(
                message.topic,
                message.data,
                message.priority,
                message.metadata
            )
        else:
            # Send to dead letter queue
            if self.config.enable_dlq:
                self.dead_letter_queue.append({
                    "message": message,
                    "error": str(error),
                    "timestamp": datetime.utcnow()
                })
                logger.warning(f"Message {message.id} sent to dead letter queue")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        error_rate = 0.0
        if self.metrics.messages_received > 0:
            error_rate = self.metrics.messages_failed / self.metrics.messages_received
        
        return {
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "messages_failed": self.metrics.messages_failed,
            "messages_retried": self.metrics.messages_retried,
            "avg_processing_time": self.metrics.avg_processing_time,
            "error_rate": error_rate,
            "queue_size": len(self.dead_letter_queue),
            "active_consumers": len(self.consumers),
            "is_running": self.is_running
        }
    
    async def cleanup(self) -> None:
        """Cleanup queue connections"""
        try:
            self.is_running = False
            
            # Cancel consumers
            for consumer in self.consumers:
                consumer.cancel()
            
            await asyncio.gather(*self.consumers, return_exceptions=True)
            
            # Close connections
            if self.redis_client:
                await self.redis_client.close()
            
            if self.rabbitmq_connection:
                await self.rabbitmq_connection.close()
            
            if self.kafka_producer:
                await self.kafka_producer.stop()
            
            logger.info("Message queue manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class EmailSequenceQueueService:
    """Service for email sequence queue operations"""
    
    def __init__(self, queue_manager: MessageQueueManager):
        self.queue_manager = queue_manager
        self.sequence_handlers: Dict[str, Callable] = {}
        
    async def publish_sequence_event(
        self,
        event_type: str,
        sequence_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Publish sequence event to queue"""
        topic = f"email_sequence.{event_type}"
        
        message_data = {
            "event_type": event_type,
            "sequence_data": sequence_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.queue_manager.publish_message(
            topic,
            message_data,
            priority
        )
    
    async def publish_email_event(
        self,
        event_type: str,
        email_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Publish email event to queue"""
        topic = f"email.{event_type}"
        
        message_data = {
            "event_type": event_type,
            "email_data": email_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.queue_manager.publish_message(
            topic,
            message_data,
            priority
        )
    
    async def subscribe_to_sequence_events(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to sequence events"""
        topic = f"email_sequence.{event_type}"
        
        async def message_handler(message: QueueMessage):
            await handler(message.data)
        
        await self.queue_manager.subscribe(topic, message_handler)
    
    async def subscribe_to_email_events(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to email events"""
        topic = f"email.{event_type}"
        
        async def message_handler(message: QueueMessage):
            await handler(message.data)
        
        await self.queue_manager.subscribe(topic, message_handler)
    
    def get_queue_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        return self.queue_manager.get_metrics() 
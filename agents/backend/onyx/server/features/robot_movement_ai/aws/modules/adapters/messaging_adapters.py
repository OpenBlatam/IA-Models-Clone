"""
Messaging Adapters
==================

Implementations of MessagingPort with different backends.
"""

import logging
from typing import Any, Dict, Optional, Callable
import os
from aws.modules.ports.messaging_port import MessagingPort

logger = logging.getLogger(__name__)


class KafkaMessagingAdapter(MessagingPort):
    """Kafka implementation of MessagingPort."""
    
    def __init__(self, bootstrap_servers: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "localhost:9092"
        )
        self._producer = None
        self._consumers: Dict[str, Any] = {}
    
    async def _get_producer(self):
        """Get Kafka producer."""
        if self._producer is None:
            try:
                from kafka import KafkaProducer
                import json
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers.split(","),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8")
                )
            except ImportError:
                logger.warning("kafka-python not installed")
            except Exception as e:
                logger.error(f"Failed to create Kafka producer: {e}")
        return self._producer
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish message to Kafka."""
        producer = await self._get_producer()
        if not producer:
            return False
        
        try:
            import json
            future = producer.send(topic, value=message, key=key.encode() if key else None)
            future.get(timeout=10)
            return True
        except Exception as e:
            logger.error(f"Kafka publish error: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to Kafka topic."""
        try:
            from kafka import KafkaConsumer
            import json
            
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_deserializer=lambda m: json.loads(m.decode("utf-8"))
            )
            
            self._consumers[topic] = consumer
            
            # Start consuming in background
            import asyncio
            asyncio.create_task(self._consume_loop(topic, consumer, handler))
            
            return True
        except Exception as e:
            logger.error(f"Kafka subscribe error: {e}")
            return False
    
    async def _consume_loop(self, topic: str, consumer: Any, handler: Callable):
        """Background consumption loop."""
        try:
            for message in consumer:
                await handler(message.value)
        except Exception as e:
            logger.error(f"Kafka consume error for {topic}: {e}")
    
    async def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from Kafka topic."""
        if topic in self._consumers:
            self._consumers[topic].close()
            del self._consumers[topic]
            return True
        return False


class RabbitMQMessagingAdapter(MessagingPort):
    """RabbitMQ implementation of MessagingPort."""
    
    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url or os.getenv(
            "RABBITMQ_URL",
            "amqp://guest:guest@localhost:5672/"
        )
        self._connection = None
        self._channel = None
    
    async def _get_connection(self):
        """Get RabbitMQ connection."""
        if self._connection is None:
            try:
                import aio_pika
                self._connection = await aio_pika.connect_robust(self.connection_url)
                self._channel = await self._connection.channel()
            except ImportError:
                logger.warning("aio_pika not installed")
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
        return self._connection
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish message to RabbitMQ."""
        connection = await self._get_connection()
        if not connection or not self._channel:
            return False
        
        try:
            import json
            exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
            await exchange.publish(
                aio_pika.Message(json.dumps(message).encode()),
                routing_key=key or ""
            )
            return True
        except Exception as e:
            logger.error(f"RabbitMQ publish error: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to RabbitMQ topic."""
        connection = await self._get_connection()
        if not connection or not self._channel:
            return False
        
        try:
            import aio_pika
            import json
            
            exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
            queue = await self._channel.declare_queue(exclusive=True)
            await queue.bind(exchange, routing_key="#")
            
            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    data = json.loads(message.body.decode())
                    await handler(data)
            
            await queue.consume(process_message)
            return True
        except Exception as e:
            logger.error(f"RabbitMQ subscribe error: {e}")
            return False
    
    async def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from RabbitMQ topic."""
        # RabbitMQ handles this automatically
        return True


class SQSMessagingAdapter(MessagingPort):
    """AWS SQS implementation of MessagingPort."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    async def _get_client(self):
        """Get SQS client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("sqs", region_name=self.region)
            except ImportError:
                logger.warning("boto3 not installed")
            except Exception as e:
                logger.error(f"Failed to create SQS client: {e}")
        return self._client
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish message to SQS."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            import json
            response = client.send_message(
                QueueUrl=topic,  # SQS queue URL
                MessageBody=json.dumps(message)
            )
            return "MessageId" in response
        except Exception as e:
            logger.error(f"SQS publish error: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to SQS topic."""
        # SQS uses polling, so this would be implemented differently
        logger.warning("SQS subscribe requires polling implementation")
        return False
    
    async def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from SQS topic."""
        return True


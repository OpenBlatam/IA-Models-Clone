"""
Event Bus Backends
Implementaciones de backends para event bus (Redis, Kafka, SQS)
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from services.event_bus import Event, EventType

logger = logging.getLogger(__name__)


class EventBusBackend(ABC):
    """Interfaz abstracta para backends de event bus"""
    
    @abstractmethod
    async def publish(self, event: Event) -> bool:
        """Publica un evento"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: EventType, handler) -> bool:
        """Suscribe un handler a un tipo de evento"""
        pass


class RedisEventBusBackend(EventBusBackend):
    """Backend de event bus usando Redis Pub/Sub"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None
        self._pubsub = None
    
    async def _get_redis(self):
        """Obtiene cliente Redis"""
        if not self._redis:
            try:
                import redis.asyncio as redis
                self._redis = await redis.from_url(self.redis_url)
                self._pubsub = self._redis.pubsub()
                logger.info("Connected to Redis for event bus")
            except ImportError:
                raise ImportError("redis package required for RedisEventBusBackend")
        return self._redis
    
    async def publish(self, event: Event) -> bool:
        """Publica evento a Redis"""
        try:
            redis = await self._get_redis()
            import json
            
            channel = f"events:{event.event_type.value}"
            message = json.dumps({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "version": event.version
            })
            
            await redis.publish(channel, message)
            logger.debug(f"Published event to Redis channel {channel}")
            return True
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")
            return False
    
    async def subscribe(self, event_type: EventType, handler) -> bool:
        """Suscribe handler a Redis channel"""
        try:
            redis = await self._get_redis()
            channel = f"events:{event_type.value}"
            
            await self._pubsub.subscribe(channel)
            logger.info(f"Subscribed to Redis channel {channel}")
            
            # Start listening in background
            import asyncio
            asyncio.create_task(self._listen(channel, handler))
            
            return True
        except Exception as e:
            logger.error(f"Error subscribing to Redis: {e}")
            return False
    
    async def _listen(self, channel: str, handler):
        """Escucha mensajes del channel"""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    import json
                    data = json.loads(message["data"])
                    event = Event(
                        event_type=EventType(data["event_type"]),
                        payload=data["payload"],
                        event_id=data["event_id"],
                        source=data.get("source", "unknown"),
                        version=data.get("version", "1.0")
                    )
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
        except Exception as e:
            logger.error(f"Error listening to Redis channel: {e}")


class KafkaEventBusBackend(EventBusBackend):
    """Backend de event bus usando Kafka"""
    
    def __init__(self, bootstrap_servers: str, topic_prefix: str = "events"):
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self._producer = None
        self._consumer = None
    
    async def _get_producer(self):
        """Obtiene producer de Kafka"""
        if not self._producer:
            try:
                from aiokafka import AIOKafkaProducer
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers
                )
                await self._producer.start()
                logger.info("Kafka producer started")
            except ImportError:
                raise ImportError("aiokafka required for KafkaEventBusBackend")
        return self._producer
    
    async def publish(self, event: Event) -> bool:
        """Publica evento a Kafka"""
        try:
            producer = await self._get_producer()
            import json
            
            topic = f"{self.topic_prefix}.{event.event_type.value}"
            message = json.dumps({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "version": event.version
            }).encode()
            
            await producer.send_and_wait(topic, message)
            logger.debug(f"Published event to Kafka topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")
            return False
    
    async def subscribe(self, event_type: EventType, handler) -> bool:
        """Suscribe handler a Kafka topic"""
        try:
            from aiokafka import AIOKafkaConsumer
            import json
            import asyncio
            
            topic = f"{self.topic_prefix}.{event_type.value}"
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"event-bus-{event_type.value}"
            )
            await consumer.start()
            
            async def consume():
                try:
                    async for msg in consumer:
                        data = json.loads(msg.value.decode())
                        event = Event(
                            event_type=EventType(data["event_type"]),
                            payload=data["payload"],
                            event_id=data["event_id"],
                            source=data.get("source", "unknown"),
                            version=data.get("version", "1.0")
                        )
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                except Exception as e:
                    logger.error(f"Error consuming from Kafka: {e}")
            
            asyncio.create_task(consume())
            logger.info(f"Subscribed to Kafka topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to Kafka: {e}")
            return False


class SQSEventBusBackend(EventBusBackend):
    """Backend de event bus usando AWS SQS"""
    
    def __init__(self, queue_url: str, region: str = "us-east-1"):
        self.queue_url = queue_url
        self.region = region
        self._sqs = None
    
    async def _get_sqs(self):
        """Obtiene cliente SQS"""
        if not self._sqs:
            try:
                from aws.services.sqs_service import SQSService
                self._sqs = SQSService(
                    queue_url=self.queue_url,
                    region_name=self.region
                )
                logger.info("Connected to SQS for event bus")
            except Exception as e:
                logger.error(f"Error connecting to SQS: {e}")
                raise
        return self._sqs
    
    async def publish(self, event: Event) -> bool:
        """Publica evento a SQS"""
        try:
            sqs = await self._get_sqs()
            import json
            
            message = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "version": event.version
            }
            
            await sqs.send_message(message)
            logger.debug(f"Published event to SQS")
            return True
        except Exception as e:
            logger.error(f"Error publishing to SQS: {e}")
            return False
    
    async def subscribe(self, event_type: EventType, handler) -> bool:
        """SQS no soporta subscriptions directas, usar Lambda/SQS integration"""
        logger.warning("SQS doesn't support direct subscriptions, use Lambda/SQS integration")
        return False


def create_event_bus_backend(backend_type: str, **kwargs) -> EventBusBackend:
    """Factory para crear backends de event bus"""
    if backend_type == "redis":
        return RedisEventBusBackend(redis_url=kwargs.get("redis_url"))
    elif backend_type == "kafka":
        return KafkaEventBusBackend(
            bootstrap_servers=kwargs.get("bootstrap_servers"),
            topic_prefix=kwargs.get("topic_prefix", "events")
        )
    elif backend_type == "sqs":
        return SQSEventBusBackend(
            queue_url=kwargs.get("queue_url"),
            region=kwargs.get("region", "us-east-1")
        )
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
















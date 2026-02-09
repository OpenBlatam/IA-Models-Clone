"""
Kafka Configuration for Event-Driven Architecture
=================================================

Handles inter-service communication using Kafka.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
).split(",")

KAFKA_TOPIC_PREFIX = os.getenv("KAFKA_TOPIC_PREFIX", "robot-movement-ai")


class KafkaEventProducer:
    """Kafka event producer for publishing events."""
    
    def __init__(self, bootstrap_servers: list = None):
        self.bootstrap_servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
        self.producer = None
        self._connect()
    
    def _connect(self):
        """Connect to Kafka."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True,
            )
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def publish(self, topic: str, event: Dict[str, Any], key: Optional[str] = None):
        """
        Publish event to Kafka topic.
        
        Args:
            topic: Kafka topic name
            event: Event data
            key: Optional partition key
        """
        if not self.producer:
            self._connect()
        
        full_topic = f"{KAFKA_TOPIC_PREFIX}.{topic}"
        
        try:
            future = self.producer.send(full_topic, value=event, key=key)
            # Wait for send to complete
            record_metadata = future.get(timeout=10)
            
            logger.info(
                f"Event published to {full_topic}",
                extra={
                    "topic": full_topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset,
                }
            )
            
            return record_metadata
            
        except KafkaError as e:
            logger.error(f"Failed to publish event to {full_topic}: {e}")
            raise
    
    def close(self):
        """Close producer connection."""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


class KafkaEventConsumer:
    """Kafka event consumer for subscribing to events."""
    
    def __init__(
        self,
        topics: list,
        group_id: str,
        bootstrap_servers: list = None,
        auto_offset_reset: str = "latest"
    ):
        self.topics = [f"{KAFKA_TOPIC_PREFIX}.{topic}" for topic in topics]
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
        self.auto_offset_reset = auto_offset_reset
        self.consumer = None
        self._connect()
    
    def _connect(self):
        """Connect to Kafka."""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
            )
            logger.info(
                f"Connected to Kafka consumer: {self.topics}, group: {self.group_id}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Kafka consumer: {e}")
            raise
    
    def consume(self, handler: callable, timeout_ms: int = 1000):
        """
        Consume events and call handler.
        
        Args:
            handler: Function to handle events
            timeout_ms: Timeout in milliseconds
        """
        if not self.consumer:
            self._connect()
        
        try:
            for message in self.consumer:
                try:
                    handler(message.topic, message.key, message.value)
                except Exception as e:
                    logger.error(
                        f"Error handling message from {message.topic}: {e}",
                        exc_info=True
                    )
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
            raise
    
    def close(self):
        """Close consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


# Event types
class EventTypes:
    """Event type constants."""
    TRAJECTORY_OPTIMIZED = "trajectory.optimized"
    MOVEMENT_STARTED = "movement.started"
    MOVEMENT_COMPLETED = "movement.completed"
    MOVEMENT_FAILED = "movement.failed"
    COLLISION_DETECTED = "collision.detected"
    ROBOT_CONNECTED = "robot.connected"
    ROBOT_DISCONNECTED = "robot.disconnected"
    HEALTH_CHECK = "health.check"


# Global producer instance
_producer: Optional[KafkaEventProducer] = None


def get_producer() -> KafkaEventProducer:
    """Get or create global Kafka producer."""
    global _producer
    if _producer is None:
        _producer = KafkaEventProducer()
    return _producer


def publish_event(event_type: str, data: Dict[str, Any], key: Optional[str] = None):
    """
    Publish event to Kafka.
    
    Args:
        event_type: Event type (from EventTypes)
        data: Event data
        key: Optional partition key
    """
    producer = get_producer()
    topic = event_type.split(".")[0]  # Extract topic from event type
    event = {
        "type": event_type,
        "data": data,
        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None,
    }
    return producer.publish(topic, event, key)
















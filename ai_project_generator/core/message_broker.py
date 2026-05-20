"""
Message Broker Integration - Integración con message brokers
============================================================

Integración con message brokers (RabbitMQ, Kafka, Redis) para
arquitectura event-driven siguiendo mejores prácticas de microservicios.
"""

import logging
import json
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

from .microservices_config import get_microservices_config, MessageBrokerType

logger = logging.getLogger(__name__)

# RabbitMQ
try:
    import pika
    from pika.adapters.blocking_connection import BlockingConnection
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

# Kafka
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Redis (ya tenemos el cliente)
from .redis_client import get_redis_client


class MessageBroker:
    """Cliente para message brokers"""
    
    def __init__(self, broker_type: Optional[MessageBrokerType] = None):
        config = get_microservices_config()
        self.broker_type = broker_type or config.message_broker_type
        self.broker_url = config.message_broker_url
        self.username = config.message_broker_username
        self.password = config.message_broker_password
        
        self.rabbitmq_connection = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.redis_client = None
        
        self._initialize_broker()
    
    def _initialize_broker(self):
        """Inicializa el broker según el tipo"""
        if self.broker_type == MessageBrokerType.RABBITMQ:
            self._init_rabbitmq()
        elif self.broker_type == MessageBrokerType.KAFKA:
            self._init_kafka()
        elif self.broker_type == MessageBrokerType.REDIS:
            self.redis_client = get_redis_client()
        else:
            logger.warning(f"Message broker type {self.broker_type} not configured")
    
    def _init_rabbitmq(self):
        """Inicializa RabbitMQ"""
        if not RABBITMQ_AVAILABLE:
            logger.warning("RabbitMQ not available. Install with: pip install pika")
            return
        
        try:
            credentials = None
            if self.username and self.password:
                credentials = pika.PlainCredentials(self.username, self.password)
            
            parameters = pika.ConnectionParameters(
                host=self.broker_url or "localhost",
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            
            self.rabbitmq_connection = BlockingConnection(parameters)
            logger.info("RabbitMQ connection established")
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ: {e}")
    
    def _init_kafka(self):
        """Inicializa Kafka"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available. Install with: pip install kafka-python")
            return
        
        try:
            bootstrap_servers = self.broker_url or "localhost:9092"
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> bool:
        """
        Publica un mensaje en el broker.
        
        Args:
            topic: Tópico/canal donde publicar
            message: Mensaje a publicar
            routing_key: Routing key (para RabbitMQ)
        
        Returns:
            True si se publicó exitosamente
        """
        try:
            if self.broker_type == MessageBrokerType.RABBITMQ:
                return await self._publish_rabbitmq(topic, message, routing_key)
            elif self.broker_type == MessageBrokerType.KAFKA:
                return await self._publish_kafka(topic, message)
            elif self.broker_type == MessageBrokerType.REDIS:
                return await self._publish_redis(topic, message)
            else:
                logger.warning("No message broker configured")
                return False
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def _publish_rabbitmq(
        self,
        exchange: str,
        message: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> bool:
        """Publica en RabbitMQ"""
        if not self.rabbitmq_connection or self.rabbitmq_connection.is_closed:
            self._init_rabbitmq()
        
        if not self.rabbitmq_connection:
            return False
        
        try:
            channel = self.rabbitmq_connection.channel()
            channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
            
            channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key or "",
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                )
            )
            
            logger.debug(f"Published message to RabbitMQ exchange {exchange}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to RabbitMQ: {e}")
            return False
    
    async def _publish_kafka(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publica en Kafka"""
        if not self.kafka_producer:
            self._init_kafka()
        
        if not self.kafka_producer:
            return False
        
        try:
            future = self.kafka_producer.send(topic, message)
            future.get(timeout=10)  # Wait for send to complete
            logger.debug(f"Published message to Kafka topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")
            return False
    
    async def _publish_redis(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publica en Redis Pub/Sub"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.publish(channel, message)
            logger.debug(f"Published message to Redis channel {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
            return False
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        routing_key: Optional[str] = None
    ):
        """
        Suscribe a un tópico y ejecuta callback cuando llegan mensajes.
        
        Args:
            topic: Tópico al que suscribirse
            callback: Función a ejecutar cuando llega un mensaje
            routing_key: Routing key (para RabbitMQ)
        """
        try:
            if self.broker_type == MessageBrokerType.RABBITMQ:
                await self._subscribe_rabbitmq(topic, callback, routing_key)
            elif self.broker_type == MessageBrokerType.KAFKA:
                await self._subscribe_kafka(topic, callback)
            elif self.broker_type == MessageBrokerType.REDIS:
                await self._subscribe_redis(topic, callback)
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
    
    async def _subscribe_rabbitmq(
        self,
        exchange: str,
        callback: Callable,
        routing_key: Optional[str] = None
    ):
        """Suscribe a RabbitMQ"""
        if not self.rabbitmq_connection:
            self._init_rabbitmq()
        
        if not self.rabbitmq_connection:
            return
        
        channel = self.rabbitmq_connection.channel()
        channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
        
        # Crear cola temporal
        result = channel.queue_declare(queue="", exclusive=True)
        queue_name = result.method.queue
        
        channel.queue_bind(
            exchange=exchange,
            queue=queue_name,
            routing_key=routing_key or "#"
        )
        
        def on_message(ch, method, properties, body):
            try:
                message = json.loads(body)
                callback(message)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing RabbitMQ message: {e}")
        
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message
        )
        
        channel.start_consuming()
    
    async def _subscribe_kafka(self, topic: str, callback: Callable):
        """Suscribe a Kafka"""
        if not KAFKA_AVAILABLE:
            return
        
        bootstrap_servers = self.broker_url or "localhost:9092"
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[bootstrap_servers],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True
        )
        
        for message in consumer:
            try:
                callback(message.value)
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
    
    async def _subscribe_redis(self, channel: str, callback: Callable):
        """Suscribe a Redis Pub/Sub"""
        if not self.redis_client:
            return
        
        async for message in self.redis_client.subscribe(channel):
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error processing Redis message: {e}")
    
    def close(self):
        """Cierra conexiones"""
        if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
            self.rabbitmq_connection.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()


# Instancia global
_message_broker: Optional[MessageBroker] = None


def get_message_broker() -> MessageBroker:
    """Obtiene instancia global de message broker"""
    global _message_broker
    if _message_broker is None:
        _message_broker = MessageBroker()
    return _message_broker
















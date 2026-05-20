"""
Message Broker - Sistema de mensajería para arquitectura event-driven
======================================================================

Soporta:
- RabbitMQ
- Apache Kafka
- Redis Pub/Sub
"""

import logging
import json
from typing import Callable, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Tipo de message broker"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"


@dataclass
class Message:
    """Mensaje del broker"""
    topic: str
    payload: Dict
    headers: Optional[Dict] = None
    message_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte el mensaje a diccionario"""
        return {
            "topic": self.topic,
            "payload": self.payload,
            "headers": self.headers or {},
            "message_id": self.message_id,
            "timestamp": self.timestamp or datetime.utcnow().isoformat()
        }


class RabbitMQBroker:
    """Broker RabbitMQ"""
    
    def __init__(self, connection_url: str = "amqp://guest:guest@localhost:5672/"):
        self.connection_url = connection_url
        self.connection = None
        self.channel = None
        self.consumers: Dict[str, List[Callable]] = {}
        self._setup()
    
    def _setup(self):
        """Configura RabbitMQ"""
        try:
            import pika
            
            self.connection = pika.BlockingConnection(
                pika.URLParameters(self.connection_url)
            )
            self.channel = self.connection.channel()
            
            logger.info("RabbitMQ connected successfully")
            
        except ImportError:
            logger.warning("pika not available. Install with: pip install pika")
            self.connection = None
            self.channel = None
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self.connection = None
            self.channel = None
    
    def publish(self, topic: str, message: Dict, exchange: str = ""):
        """Publica un mensaje"""
        if not self.channel:
            raise RuntimeError("RabbitMQ not connected")
        
        # Declarar queue
        self.channel.queue_declare(queue=topic, durable=True)
        
        # Publicar mensaje
        self.channel.basic_publish(
            exchange=exchange,
            routing_key=topic,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )
        
        logger.info(f"Message published to {topic}")
    
    def subscribe(self, topic: str, callback: Callable):
        """Suscribe un callback a un topic"""
        if not self.channel:
            raise RuntimeError("RabbitMQ not connected")
        
        # Declarar queue
        self.channel.queue_declare(queue=topic, durable=True)
        
        # Registrar callback
        if topic not in self.consumers:
            self.consumers[topic] = []
        
        self.consumers[topic].append(callback)
        
        # Configurar consumer
        def on_message(ch, method, properties, body):
            try:
                message_data = json.loads(body)
                callback(message_data)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        self.channel.basic_consume(
            queue=topic,
            on_message_callback=on_message
        )
        
        logger.info(f"Subscribed to {topic}")
    
    def start_consuming(self):
        """Inicia el consumo de mensajes"""
        if not self.channel:
            raise RuntimeError("RabbitMQ not connected")
        
        logger.info("Starting RabbitMQ consumer...")
        self.channel.start_consuming()
    
    def close(self):
        """Cierra la conexión"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")


class KafkaBroker:
    """Broker Apache Kafka"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
        self.consumers: Dict[str, List[Callable]] = {}
        self._setup()
    
    def _setup(self):
        """Configura Kafka"""
        try:
            from kafka import KafkaProducer, KafkaConsumer
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            logger.info("Kafka producer configured successfully")
            
        except ImportError:
            logger.warning("kafka-python not available. Install with: pip install kafka-python")
            self.producer = None
        except Exception as e:
            logger.error(f"Failed to setup Kafka: {e}")
            self.producer = None
    
    def publish(self, topic: str, message: Dict):
        """Publica un mensaje"""
        if not self.producer:
            raise RuntimeError("Kafka not configured")
        
        future = self.producer.send(topic, message)
        future.get(timeout=10)  # Wait for send to complete
        
        logger.info(f"Message published to Kafka topic: {topic}")
    
    def subscribe(self, topic: str, callback: Callable, group_id: str = "default"):
        """Suscribe un callback a un topic"""
        try:
            from kafka import KafkaConsumer
            
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True
            )
            
            # Registrar callback
            if topic not in self.consumers:
                self.consumers[topic] = []
            self.consumers[topic].append(callback)
            
            # Iniciar consumo en background
            import threading
            def consume():
                for message in consumer:
                    try:
                        callback(message.value)
                    except Exception as e:
                        logger.error(f"Error processing Kafka message: {e}", exc_info=True)
            
            thread = threading.Thread(target=consume, daemon=True)
            thread.start()
            
            logger.info(f"Subscribed to Kafka topic: {topic}")
            
        except ImportError:
            raise RuntimeError("kafka-python not available")
        except Exception as e:
            logger.error(f"Failed to subscribe to Kafka: {e}")
            raise
    
    def close(self):
        """Cierra las conexiones"""
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
        logger.info("Kafka connections closed")


class RedisPubSubBroker:
    """Broker Redis Pub/Sub"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.consumers: Dict[str, List[Callable]] = {}
        self._setup()
    
    def _setup(self):
        """Configura Redis Pub/Sub"""
        try:
            import redis
            
            self.redis_client = redis.from_url(self.redis_url)
            self.pubsub = self.redis_client.pubsub()
            
            logger.info("Redis Pub/Sub configured successfully")
            
        except ImportError:
            logger.warning("redis not available. Install with: pip install redis")
            self.redis_client = None
            self.pubsub = None
        except Exception as e:
            logger.error(f"Failed to setup Redis Pub/Sub: {e}")
            self.redis_client = None
            self.pubsub = None
    
    def publish(self, topic: str, message: Dict):
        """Publica un mensaje"""
        if not self.redis_client:
            raise RuntimeError("Redis not connected")
        
        message_str = json.dumps(message)
        self.redis_client.publish(topic, message_str)
        
        logger.info(f"Message published to Redis channel: {topic}")
    
    def subscribe(self, topic: str, callback: Callable):
        """Suscribe un callback a un topic"""
        if not self.pubsub:
            raise RuntimeError("Redis not connected")
        
        # Registrar callback
        if topic not in self.consumers:
            self.consumers[topic] = []
        self.consumers[topic].append(callback)
        
        # Suscribirse al canal
        self.pubsub.subscribe(topic)
        
        # Procesar mensajes en background
        import threading
        def process_messages():
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        message_data = json.loads(message['data'])
                        callback(message_data)
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}", exc_info=True)
        
        thread = threading.Thread(target=process_messages, daemon=True)
        thread.start()
        
        logger.info(f"Subscribed to Redis channel: {topic}")
    
    def close(self):
        """Cierra la conexión"""
        if self.pubsub:
            self.pubsub.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Redis Pub/Sub connection closed")


class MessageBrokerManager:
    """Gestor de message brokers"""
    
    def __init__(self, broker_type: BrokerType = BrokerType.REDIS,
                 connection_url: Optional[str] = None):
        self.broker_type = broker_type
        self.broker: Optional[Any] = None
        
        if broker_type == BrokerType.RABBITMQ:
            url = connection_url or "amqp://guest:guest@localhost:5672/"
            self.broker = RabbitMQBroker(connection_url=url)
        elif broker_type == BrokerType.KAFKA:
            servers = connection_url or "localhost:9092"
            self.broker = KafkaBroker(bootstrap_servers=servers)
        elif broker_type == BrokerType.REDIS:
            url = connection_url or "redis://localhost:6379/0"
            self.broker = RedisPubSubBroker(redis_url=url)
    
    def publish(self, topic: str, message: Dict):
        """Publica un mensaje"""
        if not self.broker:
            raise RuntimeError("Broker not configured")
        self.broker.publish(topic, message)
    
    def subscribe(self, topic: str, callback: Callable):
        """Suscribe un callback a un topic"""
        if not self.broker:
            raise RuntimeError("Broker not configured")
        self.broker.subscribe(topic, callback)
    
    def close(self):
        """Cierra el broker"""
        if self.broker:
            self.broker.close()





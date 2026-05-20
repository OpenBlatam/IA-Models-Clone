"""
Message Broker - Integración con RabbitMQ y Kafka
=================================================

Sistema de mensajería para comunicación entre servicios.
"""

import os
import logging
import json
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Tipos de message brokers soportados."""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    NATS = "nats"
    MEMORY = "memory"  # Para desarrollo/testing


class MessageBroker:
    """
    Clase base para message brokers.
    
    Proporciona interfaz común para diferentes brokers.
    """
    
    def __init__(self, broker_type: BrokerType):
        self.broker_type = broker_type
        self.connected = False
    
    async def connect(self) -> None:
        """Conectar al broker."""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Desconectar del broker."""
        raise NotImplementedError
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publicar mensaje en un topic."""
        raise NotImplementedError
    
    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Suscribirse a un topic."""
        raise NotImplementedError
    
    async def consume(self, topic: str, callback: Callable) -> None:
        """Consumir mensajes de un topic."""
        raise NotImplementedError


class RabbitMQBroker(MessageBroker):
    """Implementación para RabbitMQ."""
    
    def __init__(self, url: Optional[str] = None):
        super().__init__(BrokerType.RABBITMQ)
        self.url = url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        self.connection = None
        self.channel = None
    
    async def connect(self) -> None:
        """Conectar a RabbitMQ."""
        try:
            import aio_pika
            
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            self.connected = True
            logger.info("Connected to RabbitMQ")
        
        except ImportError:
            raise ImportError("aio-pika is required for RabbitMQ support")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Desconectar de RabbitMQ."""
        if self.connection:
            await self.connection.close()
            self.connected = False
            logger.info("Disconnected from RabbitMQ")
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publicar mensaje en RabbitMQ."""
        if not self.connected:
            await self.connect()
        
        exchange = await self.channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
        message_body = json.dumps(message).encode()
        
        await exchange.publish(
            aio_pika.Message(message_body),
            routing_key=topic
        )
        
        logger.debug(f"Published message to {topic}")
    
    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Suscribirse a un topic en RabbitMQ."""
        if not self.connected:
            await self.connect()
        
        exchange = await self.channel.declare_exchange(topic, aio_pika.ExchangeType.TOPIC)
        queue = await self.channel.declare_queue(exclusive=True)
        await queue.bind(exchange, routing_key=topic)
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    await callback(body)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        await queue.consume(process_message)
        logger.info(f"Subscribed to {topic}")


class KafkaBroker(MessageBroker):
    """Implementación para Kafka."""
    
    def __init__(self, bootstrap_servers: Optional[str] = None):
        super().__init__(BrokerType.KAFKA)
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "localhost:9092"
        )
        self.producer = None
        self.consumer = None
    
    async def connect(self) -> None:
        """Conectar a Kafka."""
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers
            )
            await self.producer.start()
            self.connected = True
            logger.info("Connected to Kafka")
        
        except ImportError:
            raise ImportError("aiokafka is required for Kafka support")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Desconectar de Kafka."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        self.connected = False
        logger.info("Disconnected from Kafka")
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publicar mensaje en Kafka."""
        if not self.connected:
            await self.connect()
        
        message_bytes = json.dumps(message).encode()
        await self.producer.send_and_wait(topic, message_bytes)
        logger.debug(f"Published message to {topic}")
    
    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Suscribirse a un topic en Kafka."""
        try:
            from aiokafka import AIOKafkaConsumer
            
            self.consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=os.getenv("KAFKA_GROUP_ID", "cursor-agent-group")
            )
            await self.consumer.start()
            
            async for msg in self.consumer:
                try:
                    message = json.loads(msg.value.decode())
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except ImportError:
            raise ImportError("aiokafka is required for Kafka support")


class MemoryBroker(MessageBroker):
    """Broker en memoria para desarrollo/testing."""
    
    def __init__(self):
        super().__init__(BrokerType.MEMORY)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
    
    async def connect(self) -> None:
        """Conectar (no-op para memory broker)."""
        self.connected = True
        logger.info("Memory broker connected")
    
    async def disconnect(self) -> None:
        """Desconectar (no-op para memory broker)."""
        self.connected = False
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publicar mensaje en memoria."""
        if topic not in self.messages:
            self.messages[topic] = []
        
        self.messages[topic].append(message)
        
        # Notificar suscriptores
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
        
        logger.debug(f"Published message to {topic} (memory)")
    
    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Suscribirse a un topic en memoria."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        self.subscribers[topic].append(callback)
        logger.info(f"Subscribed to {topic} (memory)")


def create_broker(broker_type: Optional[str] = None) -> MessageBroker:
    """
    Crear instancia de message broker.
    
    Args:
        broker_type: Tipo de broker ("rabbitmq", "kafka", "nats", "memory").
                    Si None, se detecta automáticamente.
    
    Returns:
        Instancia de MessageBroker.
    """
    if broker_type is None:
        broker_type = os.getenv("MESSAGE_BROKER_TYPE", "memory")
    
    broker_type_enum = BrokerType(broker_type.lower())
    
    if broker_type_enum == BrokerType.RABBITMQ:
        return RabbitMQBroker()
    elif broker_type_enum == BrokerType.KAFKA:
        return KafkaBroker()
    elif broker_type_enum == BrokerType.MEMORY:
        return MemoryBroker()
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}")


# Eventos estándar del sistema
class SystemEvents:
    """Eventos del sistema."""
    TASK_CREATED = "task.created"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"





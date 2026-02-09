"""
Kafka Messaging Plugin
======================
"""

import logging
from typing import Dict, Any, Optional
from aws.core.interfaces import MessagingPlugin

logger = logging.getLogger(__name__)


class KafkaMessagingPlugin(MessagingPlugin):
    """Kafka messaging plugin."""
    
    def __init__(self):
        self._producer = None
        self._initialized = False
    
    def get_name(self) -> str:
        return "kafka"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        messaging_config = config.get("messaging", {})
        return messaging_config.get("enable_kafka", False)
    
    def _initialize(self, config: Dict[str, Any]):
        """Initialize Kafka producer."""
        if self._initialized:
            return
        
        try:
            from aws.messaging.kafka_config import KafkaEventProducer
            
            messaging_config = config.get("messaging", {})
            bootstrap_servers = messaging_config.get("kafka_bootstrap_servers")
            
            if bootstrap_servers:
                self._producer = KafkaEventProducer(
                    bootstrap_servers=bootstrap_servers.split(",")
                )
                self._initialized = True
                logger.info("Kafka messaging plugin initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
    
    def publish(self, event_type: str, data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish event to Kafka."""
        if not self._initialized:
            return False
        
        try:
            topic = event_type.split(".")[0]
            event = {
                "type": event_type,
                "data": data,
            }
            self._producer.publish(topic, event, key)
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def subscribe(self, topic: str, handler: callable) -> bool:
        """Subscribe to Kafka topic."""
        # Implementation for consumer
        logger.warning("Kafka subscribe not yet implemented")
        return False
















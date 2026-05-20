from typing import Dict, Any
import logging

from ...domain.interfaces import IEventPublisher

logger = logging.getLogger(__name__)


class EventPublisherAdapter(IEventPublisher):
    
    def __init__(self, message_broker):
        self.broker = message_broker
    
    async def publish(self, event_type: str, payload: Dict[str, Any]) -> bool:
        try:
            from utils.message_broker import EventPublisher
            publisher = EventPublisher(self.broker)
            await publisher.publish_event(event_type, payload)
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
















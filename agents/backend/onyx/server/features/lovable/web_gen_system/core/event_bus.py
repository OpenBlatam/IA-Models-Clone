import logging
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)

class EventBus:
    """
    A simple Event Bus for decoupled communication between agents.
    Allows agents to publish events and subscribe to topics.
    """
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.history: List[Dict[str, Any]] = []

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe a callback function to a specific topic.
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        logger.info(f"Subscribed to topic: {topic}")

    def publish(self, topic: str, data: Dict[str, Any]):
        """
        Publish an event to a topic. All subscribers will be notified.
        """
        event = {"topic": topic, "data": data}
        self.history.append(event)
        
        logger.info(f"Event Published: {topic}")
        
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber for topic {topic}: {e}")

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

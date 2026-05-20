"""
Event Bus Module
Async event-driven messaging system.
Supports local events and MQTT for distributed systems.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Optional MQTT import
try:
    import asyncio_mqtt as aiomqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.info("asyncio-mqtt not available. MQTT disabled.")


class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Event data structure."""
    topic: str
    data: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    id: str = field(default_factory=lambda: str(datetime.now().timestamp()))


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Async event bus for pub/sub messaging.
    Supports local events and optional MQTT for distributed messaging.
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._event_history: List[Event] = []
        self._max_history = 1000
        logger.info("EventBus initialized")
    
    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe to a topic."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)
        logger.debug(f"Subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Unsubscribe from a topic."""
        if topic in self._handlers and handler in self._handlers[topic]:
            self._handlers[topic].remove(handler)
            logger.debug(f"Unsubscribed from topic: {topic}")
    
    async def publish(
        self,
        topic: str,
        data: Any,
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "system"
    ) -> None:
        """Publish an event."""
        event = Event(
            topic=topic,
            data=data,
            priority=priority,
            source=source
        )
        await self._event_queue.put(event)
        logger.debug(f"Published event to: {topic}")
    
    async def publish_and_wait(
        self,
        topic: str,
        data: Any,
        timeout: float = 30.0
    ) -> None:
        """Publish event and wait for processing."""
        await self.publish(topic, data)
        # Wait for queue to empty or timeout
        try:
            await asyncio.wait_for(self._event_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Event processing timeout for topic: {topic}")
    
    async def start(self) -> None:
        """Start the event processor."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")
    
    async def stop(self) -> None:
        """Stop the event processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("EventBus stopped")
    
    async def _process_events(self) -> None:
        """Process events from queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                await self._dispatch_event(event)
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to handlers."""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # Get handlers for exact topic and wildcards
        handlers = self._handlers.get(event.topic, [])
        handlers.extend(self._handlers.get("*", []))
        
        # Sort by priority (highest first)
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.topic}: {e}")
    
    def get_history(self, topic: str = None, limit: int = 100) -> List[Event]:
        """Get event history."""
        history = self._event_history
        if topic:
            history = [e for e in history if e.topic == topic]
        return history[-limit:]
    
    def get_topics(self) -> List[str]:
        """Get list of subscribed topics."""
        return list(self._handlers.keys())


class MQTTBridge:
    """
    MQTT bridge for distributed event messaging.
    Connects EventBus to MQTT broker.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        host: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "unified_ai"
    ):
        if not MQTT_AVAILABLE:
            raise ImportError("asyncio-mqtt required for MQTT support")
        
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.topic_prefix = topic_prefix
        self._client: Optional[aiomqtt.Client] = None
        self._running = False
    
    async def connect(self) -> None:
        """Connect to MQTT broker."""
        self._client = aiomqtt.Client(self.host, self.port)
        await self._client.__aenter__()
        self._running = True
        logger.info(f"Connected to MQTT broker: {self.host}:{self.port}")
    
    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self._running = False
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
        logger.info("Disconnected from MQTT broker")
    
    async def publish(self, topic: str, payload: Any) -> None:
        """Publish to MQTT."""
        if not self._client:
            raise RuntimeError("Not connected to MQTT broker")
        
        import json
        mqtt_topic = f"{self.topic_prefix}/{topic}"
        message = json.dumps(payload, default=str)
        await self._client.publish(mqtt_topic, message)
    
    async def subscribe(self, topic: str) -> None:
        """Subscribe to MQTT topic and forward to EventBus."""
        if not self._client:
            raise RuntimeError("Not connected to MQTT broker")
        
        mqtt_topic = f"{self.topic_prefix}/{topic}"
        await self._client.subscribe(mqtt_topic)

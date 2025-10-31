"""
Inter-service communication system.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for inter-service communication."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class Message:
    """Message structure for inter-service communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    source: str = ""
    target: str = ""
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class MessageBus:
    """Message bus for inter-service communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.request_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._message_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the message bus."""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_messages())
        logger.info("Message bus started")
    
    async def stop(self) -> None:
        """Stop the message bus."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message bus stopped")
    
    async def publish(self, message: Message) -> None:
        """Publish a message to the bus."""
        await self._message_queue.put(message)
    
    async def subscribe(self, topic: str, handler: Callable) -> None:
        """Subscribe to a topic."""
        async with self._lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(handler)
            logger.info(f"Subscribed to topic: {topic}")
    
    async def unsubscribe(self, topic: str, handler: Callable) -> None:
        """Unsubscribe from a topic."""
        async with self._lock:
            if topic in self.subscribers and handler in self.subscribers[topic]:
                self.subscribers[topic].remove(handler)
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
                logger.info(f"Unsubscribed from topic: {topic}")
    
    async def register_request_handler(self, endpoint: str, handler: Callable) -> None:
        """Register a request handler for an endpoint."""
        self.request_handlers[endpoint] = handler
        logger.info(f"Registered request handler for endpoint: {endpoint}")
    
    async def send_request(
        self, 
        target: str, 
        endpoint: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send a request and wait for response."""
        request_id = str(uuid.uuid4())
        request = Message(
            type=MessageType.REQUEST,
            source="message_bus",
            target=target,
            topic=endpoint,
            payload=payload,
            correlation_id=request_id
        )
        
        # Create response future
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future
        
        # Send request
        await self.publish(request)
        
        try:
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response.payload
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {target}/{endpoint} timed out")
        finally:
            # Clean up
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
    
    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, message: Message) -> None:
        """Handle a received message."""
        try:
            if message.type == MessageType.REQUEST:
                await self._handle_request(message)
            elif message.type == MessageType.RESPONSE:
                await self._handle_response(message)
            elif message.type == MessageType.EVENT:
                await self._handle_event(message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")
    
    async def _handle_request(self, message: Message) -> None:
        """Handle a request message."""
        endpoint = message.topic
        
        if endpoint not in self.request_handlers:
            # Send error response
            error_response = Message(
                type=MessageType.ERROR,
                source="message_bus",
                target=message.source,
                payload={"error": f"Unknown endpoint: {endpoint}"},
                correlation_id=message.correlation_id
            )
            await self.publish(error_response)
            return
        
        try:
            handler = self.request_handlers[endpoint]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message.payload)
            else:
                result = handler(message.payload)
            
            # Send response
            response = Message(
                type=MessageType.RESPONSE,
                source="message_bus",
                target=message.source,
                payload=result,
                correlation_id=message.correlation_id
            )
            await self.publish(response)
            
        except Exception as e:
            # Send error response
            error_response = Message(
                type=MessageType.ERROR,
                source="message_bus",
                target=message.source,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
            await self.publish(error_response)
    
    async def _handle_response(self, message: Message) -> None:
        """Handle a response message."""
        if message.correlation_id and message.correlation_id in self._pending_requests:
            future = self._pending_requests[message.correlation_id]
            if not future.done():
                future.set_result(message)
    
    async def _handle_event(self, message: Message) -> None:
        """Handle an event message."""
        topic = message.topic
        
        if topic in self.subscribers:
            for handler in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message.payload)
                    else:
                        handler(message.payload)
                except Exception as e:
                    logger.error(f"Error in event handler for topic {topic}: {e}")


class EventPublisher:
    """Publisher for events."""
    
    def __init__(self, message_bus: MessageBus, service_name: str):
        self.message_bus = message_bus
        self.service_name = service_name
    
    async def publish_event(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish an event."""
        message = Message(
            type=MessageType.EVENT,
            source=self.service_name,
            topic=topic,
            payload=payload
        )
        await self.message_bus.publish(message)
        logger.debug(f"Published event to topic {topic}")


class EventSubscriber:
    """Subscriber for events."""
    
    def __init__(self, message_bus: MessageBus, service_name: str):
        self.message_bus = message_bus
        self.service_name = service_name
        self.handlers: Dict[str, Callable] = {}
    
    async def subscribe(self, topic: str, handler: Callable) -> None:
        """Subscribe to an event topic."""
        self.handlers[topic] = handler
        await self.message_bus.subscribe(topic, handler)
        logger.info(f"Subscribed to event topic: {topic}")
    
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from an event topic."""
        if topic in self.handlers:
            handler = self.handlers[topic]
            await self.message_bus.unsubscribe(topic, handler)
            del self.handlers[topic]
            logger.info(f"Unsubscribed from event topic: {topic}")


class ServiceClient:
    """Client for making requests to other services."""
    
    def __init__(self, message_bus: MessageBus, service_name: str):
        self.message_bus = message_bus
        self.service_name = service_name
        self._pending_requests: Dict[str, asyncio.Future] = {}
    
    async def call_service(
        self, 
        target_service: str, 
        endpoint: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Call a service endpoint."""
        return await self.message_bus.send_request(
            target_service, endpoint, payload, timeout
        )
    
    async def call_export_service(
        self, 
        content: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> str:
        """Call the export service."""
        response = await self.call_service(
            "export-service",
            "export",
            {"content": content, "config": config}
        )
        return response.get("task_id")
    
    async def call_quality_service(
        self, 
        content: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the quality service."""
        return await self.call_service(
            "quality-service",
            "validate",
            {"content": content, "config": config}
        )


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus





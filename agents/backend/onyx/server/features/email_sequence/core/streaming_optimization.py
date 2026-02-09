"""
Real-time Streaming Optimization for Email Sequence System

Provides advanced streaming capabilities including WebSocket support,
Server-Sent Events, and real-time analytics for the email sequence system.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import websockets
from websockets.server import serve, WebSocketServerProtocol
import aiohttp
from aiohttp import web, ClientSession
import sse_starlette
from sse_starlette.sse import EventSourceResponse

# Models
from ..models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from ..models.subscriber import Subscriber, SubscriberSegment
from ..models.template import EmailTemplate, TemplateVariable
from ..models.campaign import EmailCampaign, CampaignMetrics

logger = logging.getLogger(__name__)

# Constants
MAX_CONNECTIONS = 1000
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
HEARTBEAT_INTERVAL = 30  # seconds
CONNECTION_TIMEOUT = 300  # seconds
STREAM_BUFFER_SIZE = 1000


class StreamType(Enum):
    """Stream types"""
    WEBSOCKET = "websocket"
    SSE = "sse"  # Server-Sent Events
    HTTP_STREAM = "http_stream"
    GRPC_STREAM = "grpc_stream"


class StreamEventType(Enum):
    """Stream event types"""
    SEQUENCE_CREATED = "sequence_created"
    SEQUENCE_UPDATED = "sequence_updated"
    SEQUENCE_DELETED = "sequence_deleted"
    EMAIL_SENT = "email_sent"
    EMAIL_DELIVERED = "email_delivered"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    SUBSCRIBER_ADDED = "subscriber_added"
    SUBSCRIBER_REMOVED = "subscriber_removed"
    ANALYTICS_UPDATE = "analytics_update"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """Stream event structure"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StreamConfig:
    """Stream configuration"""
    stream_type: StreamType
    max_connections: int = MAX_CONNECTIONS
    max_message_size: int = MAX_MESSAGE_SIZE
    heartbeat_interval: int = HEARTBEAT_INTERVAL
    connection_timeout: int = CONNECTION_TIMEOUT
    stream_buffer_size: int = STREAM_BUFFER_SIZE
    enable_compression: bool = True
    enable_encryption: bool = False
    enable_metrics: bool = True
    enable_heartbeat: bool = True
    enable_reconnection: bool = True
    cors_origins: List[str] = None


@dataclass
class StreamMetrics:
    """Stream performance metrics"""
    total_connections: int = 0
    active_connections: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    avg_latency: float = 0.0
    connection_duration: float = 0.0


class StreamConnection:
    """Individual stream connection"""
    
    def __init__(self, connection_id: str, websocket: WebSocketServerProtocol = None):
        self.connection_id = connection_id
        self.websocket = websocket
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.is_active = True
        self.subscriptions: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
    async def send_event(self, event: StreamEvent) -> bool:
        """Send event to connection"""
        try:
            if self.websocket and self.is_active:
                message = json.dumps({
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data,
                    "metadata": event.metadata
                })
                
                await self.websocket.send(message)
                return True
                
        except Exception as e:
            logger.error(f"Error sending event to connection {self.connection_id}: {e}")
            self.is_active = False
            return False
    
    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if connection is expired"""
        return (datetime.utcnow() - self.last_heartbeat).total_seconds() > timeout_seconds


class RealTimeStreamManager:
    """Advanced real-time stream manager with multiple protocol support"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.connections: Dict[str, StreamConnection] = {}
        self.event_handlers: Dict[StreamEventType, List[Callable]] = defaultdict(list)
        self.event_buffer: deque = deque(maxlen=config.stream_buffer_size)
        
        # Performance tracking
        self.metrics = StreamMetrics()
        self.latency_times: List[float] = []
        
        # WebSocket server
        self.websocket_server = None
        self.http_server = None
        
        # Event broadcasting
        self.broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.is_running = False
        
        logger.info(f"Real-time Stream Manager initialized for {config.stream_type.value}")
    
    async def initialize(self) -> None:
        """Initialize streaming components"""
        try:
            if self.config.stream_type == StreamType.WEBSOCKET:
                await self._initialize_websocket_server()
            elif self.config.stream_type == StreamType.SSE:
                await self._initialize_sse_server()
            elif self.config.stream_type == StreamType.HTTP_STREAM:
                await self._initialize_http_stream_server()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Real-time stream manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize stream manager: {e}")
            raise
    
    async def _initialize_websocket_server(self) -> None:
        """Initialize WebSocket server"""
        async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
            connection_id = str(uuid.uuid4())
            connection = StreamConnection(connection_id, websocket)
            self.connections[connection_id] = connection
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
            
            try:
                async for message in websocket:
                    await self._handle_websocket_message(connection_id, message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"WebSocket connection {connection_id} closed")
            except Exception as e:
                logger.error(f"WebSocket error for connection {connection_id}: {e}")
            finally:
                await self._cleanup_connection(connection_id)
        
        self.websocket_server = await serve(
            websocket_handler,
            "localhost",
            8765,
            max_size=self.config.max_message_size
        )
        logger.info("WebSocket server started on ws://localhost:8765")
    
    async def _initialize_sse_server(self) -> None:
        """Initialize Server-Sent Events server"""
        app = web.Application()
        
        async def sse_handler(request):
            connection_id = str(uuid.uuid4())
            connection = StreamConnection(connection_id)
            self.connections[connection_id] = connection
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
            
            async def event_generator():
                try:
                    while connection.is_active:
                        # Send heartbeat
                        if self.config.enable_heartbeat:
                            heartbeat_event = StreamEvent(
                                event_id=str(uuid.uuid4()),
                                event_type=StreamEventType.HEARTBEAT,
                                timestamp=datetime.utcnow(),
                                data={"connection_id": connection_id}
                            )
                            yield {
                                "event": "heartbeat",
                                "data": json.dumps(heartbeat_event.data)
                            }
                        
                        await asyncio.sleep(self.config.heartbeat_interval)
                        
                except Exception as e:
                    logger.error(f"SSE error for connection {connection_id}: {e}")
                finally:
                    await self._cleanup_connection(connection_id)
            
            return EventSourceResponse(event_generator())
        
        app.router.add_get("/stream", sse_handler)
        self.http_server = app
        logger.info("SSE server initialized")
    
    async def _initialize_http_stream_server(self) -> None:
        """Initialize HTTP streaming server"""
        app = web.Application()
        
        async def stream_handler(request):
            connection_id = str(uuid.uuid4())
            connection = StreamConnection(connection_id)
            self.connections[connection_id] = connection
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
            
            response = web.StreamResponse(
                status=200,
                headers={
                    'Content-Type': 'text/plain',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            await response.prepare(request)
            
            try:
                while connection.is_active:
                    # Send heartbeat
                    if self.config.enable_heartbeat:
                        heartbeat_data = {
                            "type": "heartbeat",
                            "timestamp": datetime.utcnow().isoformat(),
                            "connection_id": connection_id
                        }
                        await response.write(f"data: {json.dumps(heartbeat_data)}\n\n".encode())
                    
                    await asyncio.sleep(self.config.heartbeat_interval)
                    
            except Exception as e:
                logger.error(f"HTTP stream error for connection {connection_id}: {e}")
            finally:
                await self._cleanup_connection(connection_id)
            
            return response
        
        app.router.add_get("/stream", stream_handler)
        self.http_server = app
        logger.info("HTTP stream server initialized")
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks"""
        self.is_running = True
        
        # Start event broadcasting
        asyncio.create_task(self._broadcast_events())
        
        # Start connection cleanup
        asyncio.create_task(self._cleanup_expired_connections())
        
        # Start metrics collection
        asyncio.create_task(self._collect_metrics())
    
    async def _broadcast_events(self) -> None:
        """Broadcast events to all connections"""
        while self.is_running:
            try:
                event = await asyncio.wait_for(
                    self.broadcast_queue.get(),
                    timeout=1.0
                )
                
                # Send to all active connections
                for connection_id, connection in self.connections.items():
                    if connection.is_active:
                        await connection.send_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event broadcasting error: {e}")
    
    async def _cleanup_expired_connections(self) -> None:
        """Cleanup expired connections"""
        while self.is_running:
            try:
                expired_connections = [
                    conn_id for conn_id, connection in self.connections.items()
                    if connection.is_expired(self.config.connection_timeout)
                ]
                
                for conn_id in expired_connections:
                    await self._cleanup_connection(conn_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
    
    async def _collect_metrics(self) -> None:
        """Collect stream metrics"""
        while self.is_running:
            try:
                # Update active connections count
                self.metrics.active_connections = sum(
                    1 for conn in self.connections.values()
                    if conn.is_active
                )
                
                # Calculate average latency
                if self.latency_times:
                    self.metrics.avg_latency = sum(self.latency_times) / len(self.latency_times)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _handle_websocket_message(self, connection_id: str, message: str) -> None:
        """Handle WebSocket message"""
        try:
            data = json.loads(message)
            event_type = data.get("event_type")
            
            if event_type == "subscribe":
                # Handle subscription
                topics = data.get("topics", [])
                connection = self.connections.get(connection_id)
                if connection:
                    connection.subscriptions.extend(topics)
                    
            elif event_type == "unsubscribe":
                # Handle unsubscription
                topics = data.get("topics", [])
                connection = self.connections.get(connection_id)
                if connection:
                    for topic in topics:
                        if topic in connection.subscriptions:
                            connection.subscriptions.remove(topic)
            
            self.metrics.messages_received += 1
            self.metrics.bytes_received += len(message.encode())
            
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
            self.metrics.errors += 1
    
    async def _cleanup_connection(self, connection_id: str) -> None:
        """Cleanup connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.is_active = False
            
            # Calculate connection duration
            duration = (datetime.utcnow() - connection.connected_at).total_seconds()
            self.metrics.connection_duration = duration
            
            del self.connections[connection_id]
            self.metrics.active_connections -= 1
            
            logger.info(f"Connection {connection_id} cleaned up")
    
    async def broadcast_event(self, event: StreamEvent) -> None:
        """Broadcast event to all connections"""
        try:
            # Add to event buffer
            self.event_buffer.append(event)
            
            # Add to broadcast queue
            await self.broadcast_queue.put(event)
            
            # Execute event handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await asyncio.create_task(handler(event))
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
            
            self.metrics.messages_sent += 1
            
        except Exception as e:
            logger.error(f"Event broadcasting error: {e}")
            self.metrics.errors += 1
    
    async def subscribe_to_events(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Subscribe to stream events"""
        self.event_handlers[event_type].append(handler)
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get stream metrics"""
        return {
            "total_connections": self.metrics.total_connections,
            "active_connections": self.metrics.active_connections,
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "bytes_sent": self.metrics.bytes_sent,
            "bytes_received": self.metrics.bytes_received,
            "errors": self.metrics.errors,
            "avg_latency": self.metrics.avg_latency,
            "connection_duration": self.metrics.connection_duration,
            "event_buffer_size": len(self.event_buffer),
            "broadcast_queue_size": self.broadcast_queue.qsize()
        }
    
    async def cleanup(self) -> None:
        """Cleanup stream manager"""
        try:
            self.is_running = False
            
            # Close all connections
            for connection_id in list(self.connections.keys()):
                await self._cleanup_connection(connection_id)
            
            # Stop servers
            if self.websocket_server:
                self.websocket_server.close()
            
            logger.info("Stream manager cleaned up")
            
        except Exception as e:
            logger.error(f"Stream cleanup error: {e}")


class EmailSequenceStreamService:
    """Service for email sequence streaming operations"""
    
    def __init__(self, stream_manager: RealTimeStreamManager):
        self.stream_manager = stream_manager
        self.sequence_events: Dict[str, List[StreamEvent]] = {}
        self.analytics_streams: Dict[str, AsyncGenerator] = {}
    
    async def stream_sequence_events(self, sequence_id: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a specific sequence"""
        try:
            while True:
                # Get events for sequence
                events = self.sequence_events.get(sequence_id, [])
                
                for event in events:
                    yield event
                
                # Clear processed events
                self.sequence_events[sequence_id] = []
                
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"Sequence event streaming error: {e}")
    
    async def stream_analytics(self, sequence_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream analytics for a sequence"""
        try:
            while True:
                # Generate analytics data
                analytics_data = {
                    "sequence_id": sequence_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {
                        "emails_sent": 0,
                        "emails_delivered": 0,
                        "emails_opened": 0,
                        "emails_clicked": 0,
                        "open_rate": 0.0,
                        "click_rate": 0.0
                    }
                }
                
                yield analytics_data
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"Analytics streaming error: {e}")
    
    async def publish_sequence_event(
        self,
        event_type: StreamEventType,
        sequence_id: str,
        data: Dict[str, Any]
    ) -> None:
        """Publish sequence event"""
        try:
            event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.utcnow(),
                data=data,
                metadata={"sequence_id": sequence_id}
            )
            
            # Add to sequence events
            if sequence_id not in self.sequence_events:
                self.sequence_events[sequence_id] = []
            
            self.sequence_events[sequence_id].append(event)
            
            # Broadcast event
            await self.stream_manager.broadcast_event(event)
            
        except Exception as e:
            logger.error(f"Sequence event publishing error: {e}")
    
    async def publish_email_event(
        self,
        event_type: StreamEventType,
        email_data: Dict[str, Any]
    ) -> None:
        """Publish email event"""
        try:
            event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.utcnow(),
                data=email_data
            )
            
            # Broadcast event
            await self.stream_manager.broadcast_event(event)
            
        except Exception as e:
            logger.error(f"Email event publishing error: {e}")
    
    async def subscribe_to_sequence_events(
        self,
        sequence_id: str,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Subscribe to sequence events"""
        async def sequence_event_handler(event: StreamEvent):
            if event.metadata.get("sequence_id") == sequence_id:
                await handler(event)
        
        await self.stream_manager.subscribe_to_events(
            StreamEventType.SEQUENCE_CREATED,
            sequence_event_handler
        )
        await self.stream_manager.subscribe_to_events(
            StreamEventType.SEQUENCE_UPDATED,
            sequence_event_handler
        )
        await self.stream_manager.subscribe_to_events(
            StreamEventType.SEQUENCE_DELETED,
            sequence_event_handler
        )
    
    async def subscribe_to_email_events(
        self,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Subscribe to email events"""
        await self.stream_manager.subscribe_to_events(
            StreamEventType.EMAIL_SENT,
            handler
        )
        await self.stream_manager.subscribe_to_events(
            StreamEventType.EMAIL_DELIVERED,
            handler
        )
        await self.stream_manager.subscribe_to_events(
            StreamEventType.EMAIL_OPENED,
            handler
        )
        await self.stream_manager.subscribe_to_events(
            StreamEventType.EMAIL_CLICKED,
            handler
        )
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get stream metrics"""
        return self.stream_manager.get_stream_metrics() 
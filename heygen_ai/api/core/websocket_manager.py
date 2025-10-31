from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
import uuid
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import logging
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from starlette.websockets import WebSocketState
import structlog
from typing import Any, List, Dict, Optional
"""
WebSocket Manager - Real-time Communication
Comprehensive WebSocket management for real-time video processing updates and notifications.
"""



logger = structlog.get_logger()

# =============================================================================
# WEBSOCKET MESSAGE TYPES
# =============================================================================

class MessageType(str, Enum):
    """WebSocket message types."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    VIDEO_PROGRESS = "video_progress"
    VIDEO_COMPLETE = "video_complete"
    VIDEO_ERROR = "video_error"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    NOTIFICATION = "notification"
    SYSTEM_STATUS = "system_status"


class ConnectionStatus(str, Enum):
    """WebSocket connection status."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


# =============================================================================
# WEBSOCKET MESSAGE MODELS
# =============================================================================

class WebSocketMessage:
    """Base WebSocket message model."""
    
    def __init__(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        
    """__init__ function."""
self.message_type = message_type
        self.data = data
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "type": self.message_type.value,
            "data": self.data,
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data["type"]),
            data=data["data"],
            message_id=data.get("message_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON string."""
        return cls.from_dict(json.loads(json_str))


class VideoProgressMessage(WebSocketMessage):
    """Video progress update message."""
    
    def __init__(
        self,
        video_id: str,
        progress: float,
        status: str,
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        
    """__init__ function."""
super().__init__(
            message_type=MessageType.VIDEO_PROGRESS,
            data={
                "video_id": video_id,
                "progress": progress,
                "status": status,
                "estimated_completion": None  # Will be calculated
            },
            message_id=message_id,
            timestamp=timestamp
        )


class VideoCompleteMessage(WebSocketMessage):
    """Video completion message."""
    
    def __init__(
        self,
        video_id: str,
        output_url: str,
        duration: float,
        file_size: int,
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        
    """__init__ function."""
super().__init__(
            message_type=MessageType.VIDEO_COMPLETE,
            data={
                "video_id": video_id,
                "output_url": output_url,
                "duration": duration,
                "file_size": file_size,
                "completed_at": datetime.now().isoformat()
            },
            message_id=message_id,
            timestamp=timestamp
        )


class VideoErrorMessage(WebSocketMessage):
    """Video error message."""
    
    def __init__(
        self,
        video_id: str,
        error_code: str,
        error_message: str,
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        
    """__init__ function."""
super().__init__(
            message_type=MessageType.VIDEO_ERROR,
            data={
                "video_id": video_id,
                "error_code": error_code,
                "error_message": error_message,
                "failed_at": datetime.now().isoformat()
            },
            message_id=message_id,
            timestamp=timestamp
        )


# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================

class WebSocketConnection:
    """Individual WebSocket connection with metadata."""
    
    def __init__(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        
    """__init__ function."""
self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.status = ConnectionStatus.CONNECTING
        self.connected_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.subscriptions: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.last_activity = datetime.now()
    
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return (
            self.websocket.client_state == WebSocketState.CONNECTED and
            self.status == ConnectionStatus.CONNECTED
        )
    
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is stale (no activity for timeout_seconds)."""
        return (datetime.now() - self.last_activity).total_seconds() > timeout_seconds
    
    def update_activity(self) -> Any:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def add_subscription(self, topic: str):
        """Add subscription to a topic."""
        self.subscriptions.add(topic)
    
    def remove_subscription(self, topic: str):
        """Remove subscription from a topic."""
        self.subscriptions.discard(topic)
    
    def has_subscription(self, topic: str) -> bool:
        """Check if connection is subscribed to topic."""
        return topic in self.subscriptions


class WebSocketManager:
    """Main WebSocket manager for handling connections and message broadcasting."""
    
    def __init__(
        self,
        heartbeat_interval: int = 30,
        cleanup_interval: int = 60,
        max_connections: int = 1000,
        max_subscriptions_per_connection: int = 50
    ):
        
    """__init__ function."""
self.connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = {}
        self.heartbeat_interval = heartbeat_interval
        self.cleanup_interval = cleanup_interval
        self.max_connections = max_connections
        self.max_subscriptions_per_connection = max_subscriptions_per_connection
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        
        # Event handlers
        self.on_connect_handlers: List[Callable[[WebSocketConnection], Awaitable[None]]] = []
        self.on_disconnect_handlers: List[Callable[[WebSocketConnection], Awaitable[None]]] = []
        self.on_message_handlers: List[Callable[[WebSocketConnection, WebSocketMessage], Awaitable[None]]] = []
    
    async def start(self) -> Any:
        """Start the WebSocket manager."""
        logger.info("Starting WebSocket manager")
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("WebSocket manager started")
    
    async def stop(self) -> Any:
        """Stop the WebSocket manager."""
        logger.info("Stopping WebSocket manager")
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        await self.close_all_connections()
        
        logger.info("WebSocket manager stopped")
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: Optional[str] = None
    ) -> WebSocketConnection:
        """Accept a new WebSocket connection."""
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Maximum connections reached"
            )
        
        # Accept connection
        await websocket.accept()
        
        # Create connection object
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id, user_id)
        connection.status = ConnectionStatus.CONNECTED
        
        # Store connection
        self.connections[connection_id] = connection
        self.total_connections += 1
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            message_type=MessageType.CONNECT,
            data={
                "connection_id": connection_id,
                "user_id": user_id,
                "message": "Connected successfully"
            }
        )
        await self.send_message(connection, welcome_message)
        
        # Call connect handlers
        for handler in self.on_connect_handlers:
            try:
                await handler(connection)
            except Exception as e:
                logger.error(f"Connect handler error: {e}")
        
        logger.info(f"WebSocket connected", 
                   connection_id=connection_id, 
                   user_id=user_id,
                   total_connections=len(self.connections))
        
        return connection
    
    async def disconnect(self, connection_id: str, reason: str = "Disconnected"):
        """Disconnect a WebSocket connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Update status
        connection.status = ConnectionStatus.DISCONNECTED
        
        # Remove from topic subscriptions
        for topic in list(connection.subscriptions):
            await self.unsubscribe(connection_id, topic)
        
        # Send disconnect message
        try:
            disconnect_message = WebSocketMessage(
                message_type=MessageType.DISCONNECT,
                data={"reason": reason}
            )
            await self.send_message(connection, disconnect_message)
        except Exception as e:
            logger.warning(f"Failed to send disconnect message: {e}")
        
        # Close websocket
        try:
            await connection.websocket.close()
        except Exception as e:
            logger.warning(f"Failed to close websocket: {e}")
        
        # Remove from connections
        self.connections.pop(connection_id, None)
        
        # Call disconnect handlers
        for handler in self.on_disconnect_handlers:
            try:
                await handler(connection)
            except Exception as e:
                logger.error(f"Disconnect handler error: {e}")
        
        logger.info(f"WebSocket disconnected", 
                   connection_id=connection_id,
                   reason=reason,
                   total_connections=len(self.connections))
    
    async def send_message(self, connection: WebSocketConnection, message: WebSocketMessage) -> bool:
        """Send a message to a specific connection."""
        if not connection.is_connected():
            return False
        
        try:
            await connection.websocket.send_text(message.to_json())
            connection.messages_sent += 1
            connection.update_activity()
            self.total_messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            await self.disconnect(connection.connection_id, "Send error")
            return False
    
    async def broadcast_message(self, message: WebSocketMessage, topic: Optional[str] = None):
        """Broadcast a message to all connections or topic subscribers."""
        connections_to_send = []
        
        if topic:
            # Send to topic subscribers
            subscriber_ids = self.topic_subscriptions.get(topic, set())
            connections_to_send = [
                self.connections[conn_id] for conn_id in subscriber_ids
                if conn_id in self.connections
            ]
        else:
            # Send to all connections
            connections_to_send = list(self.connections.values())
        
        # Send message to all connections
        failed_connections = []
        for connection in connections_to_send:
            if not await self.send_message(connection, message):
                failed_connections.append(connection.connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id, "Broadcast failed")
        
        logger.info(f"Broadcasted message", 
                   message_type=message.message_type.value,
                   topic=topic,
                   sent_to=len(connections_to_send),
                   failed=len(failed_connections))
    
    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic."""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        # Check subscription limit
        if len(connection.subscriptions) >= self.max_subscriptions_per_connection:
            return False
        
        # Add subscription
        connection.add_subscription(topic)
        
        # Add to topic subscribers
        if topic not in self.topic_subscriptions:
            self.topic_subscriptions[topic] = set()
        self.topic_subscriptions[topic].add(connection_id)
        
        # Send confirmation
        subscribe_message = WebSocketMessage(
            message_type=MessageType.SUBSCRIBE,
            data={"topic": topic, "status": "subscribed"}
        )
        await self.send_message(connection, subscribe_message)
        
        logger.info(f"Subscribed to topic", 
                   connection_id=connection_id,
                   topic=topic)
        
        return True
    
    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic."""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        # Remove subscription
        connection.remove_subscription(topic)
        
        # Remove from topic subscribers
        if topic in self.topic_subscriptions:
            self.topic_subscriptions[topic].discard(connection_id)
            if not self.topic_subscriptions[topic]:
                del self.topic_subscriptions[topic]
        
        # Send confirmation
        unsubscribe_message = WebSocketMessage(
            message_type=MessageType.UNSUBSCRIBE,
            data={"topic": topic, "status": "unsubscribed"}
        )
        await self.send_message(connection, unsubscribe_message)
        
        logger.info(f"Unsubscribed from topic", 
                   connection_id=connection_id,
                   topic=topic)
        
        return True
    
    async def handle_message(self, connection: WebSocketConnection, message_text: str):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            message = WebSocketMessage.from_json(message_text)
            connection.messages_received += 1
            connection.update_activity()
            self.total_messages_received += 1
            
            # Handle message based on type
            if message.message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(connection, message)
            elif message.message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(connection, message)
            elif message.message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(connection, message)
            else:
                # Call message handlers
                for handler in self.on_message_handlers:
                    try:
                        await handler(connection, message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
            error_message = WebSocketMessage(
                message_type=MessageType.NOTIFICATION,
                data={"error": "Invalid JSON message"}
            )
            await self.send_message(connection, error_message)
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def _handle_heartbeat(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Handle heartbeat message."""
        connection.last_heartbeat = datetime.now()
        
        # Send heartbeat response
        heartbeat_response = WebSocketMessage(
            message_type=MessageType.HEARTBEAT,
            data={"timestamp": datetime.now().isoformat()}
        )
        await self.send_message(connection, heartbeat_response)
    
    async def _handle_subscribe(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Handle subscribe message."""
        topic = message.data.get("topic")
        if topic:
            await self.subscribe(connection.connection_id, topic)
    
    async def _handle_unsubscribe(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Handle unsubscribe message."""
        topic = message.data.get("topic")
        if topic:
            await self.unsubscribe(connection.connection_id, topic)
    
    async def _heartbeat_loop(self) -> Any:
        """Background task for sending heartbeat messages."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all connections
                heartbeat_message = WebSocketMessage(
                    message_type=MessageType.HEARTBEAT,
                    data={"timestamp": datetime.now().isoformat()}
                )
                
                await self.broadcast_message(heartbeat_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    async def _cleanup_loop(self) -> Any:
        """Background task for cleaning up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Find stale connections
                stale_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.is_stale():
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id, "Connection timeout")
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def close_all_connections(self) -> Any:
        """Close all active connections."""
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id, "Server shutdown")
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get connection by ID."""
        return self.connections.get(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": self.total_connections,
            "active_connections": len(self.connections),
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "topic_subscriptions": len(self.topic_subscriptions),
            "connections_by_user": self._get_connections_by_user()
        }
    
    def _get_connections_by_user(self) -> Dict[str, int]:
        """Get connection count by user."""
        user_counts = {}
        for connection in self.connections.values():
            user_id = connection.user_id or "anonymous"
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        return user_counts


# =============================================================================
# WEBSOCKET EVENT HANDLERS
# =============================================================================

class WebSocketEventHandler:
    """Event handler for WebSocket events."""
    
    def __init__(self, manager: WebSocketManager):
        
    """__init__ function."""
self.manager = manager
        self._setup_handlers()
    
    def _setup_handlers(self) -> Any:
        """Setup default event handlers."""
        self.manager.on_connect_handlers.append(self._on_connect)
        self.manager.on_disconnect_handlers.append(self._on_disconnect)
        self.manager.on_message_handlers.append(self._on_message)
    
    async def _on_connect(self, connection: WebSocketConnection):
        """Handle connection event."""
        logger.info(f"Connection established", 
                   connection_id=connection.connection_id,
                   user_id=connection.user_id)
    
    async def _on_disconnect(self, connection: WebSocketConnection):
        """Handle disconnection event."""
        logger.info(f"Connection closed", 
                   connection_id=connection.connection_id,
                   user_id=connection.user_id,
                   duration=(datetime.now() - connection.connected_at).total_seconds())
    
    async def _on_message(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Handle incoming message."""
        logger.debug(f"Received message", 
                    connection_id=connection.connection_id,
                    message_type=message.message_type.value)


# =============================================================================
# VIDEO-SPECIFIC WEBSOCKET FUNCTIONS
# =============================================================================

class VideoWebSocketManager(WebSocketManager):
    """WebSocket manager specialized for video processing updates."""
    
    def __init__(self, **kwargs) -> Any:
        super().__init__(**kwargs)
        self.video_subscriptions: Dict[str, Set[str]] = {}  # video_id -> connection_ids
    
    async def subscribe_to_video(self, connection_id: str, video_id: str) -> bool:
        """Subscribe to video progress updates."""
        # Subscribe to general video topic
        await self.subscribe(connection_id, f"video:{video_id}")
        
        # Add to video subscriptions
        if video_id not in self.video_subscriptions:
            self.video_subscriptions[video_id] = set()
        self.video_subscriptions[video_id].add(connection_id)
        
        return True
    
    async def unsubscribe_from_video(self, connection_id: str, video_id: str) -> bool:
        """Unsubscribe from video progress updates."""
        # Unsubscribe from general video topic
        await self.unsubscribe(connection_id, f"video:{video_id}")
        
        # Remove from video subscriptions
        if video_id in self.video_subscriptions:
            self.video_subscriptions[video_id].discard(connection_id)
            if not self.video_subscriptions[video_id]:
                del self.video_subscriptions[video_id]
        
        return True
    
    async def broadcast_video_progress(self, video_id: str, progress: float, status: str):
        """Broadcast video progress update."""
        message = VideoProgressMessage(video_id, progress, status)
        await self.broadcast_message(message, f"video:{video_id}")
    
    async def broadcast_video_complete(self, video_id: str, output_url: str, duration: float, file_size: int):
        """Broadcast video completion."""
        message = VideoCompleteMessage(video_id, output_url, duration, file_size)
        await self.broadcast_message(message, f"video:{video_id}")
    
    async def broadcast_video_error(self, video_id: str, error_code: str, error_message: str):
        """Broadcast video error."""
        message = VideoErrorMessage(video_id, error_code, error_message)
        await self.broadcast_message(message, f"video:{video_id}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_websocket_manager() -> VideoWebSocketManager:
    """Get WebSocket manager instance."""
    return VideoWebSocketManager(
        heartbeat_interval=30,
        cleanup_interval=60,
        max_connections=1000,
        max_subscriptions_per_connection=50
    )


async def websocket_endpoint(websocket: WebSocket, user_id: Optional[str] = None):
    """WebSocket endpoint handler."""
    manager = get_websocket_manager()
    
    try:
        # Connect
        connection = await manager.connect(websocket, user_id)
        
        # Message loop
        while connection.is_connected():
            try:
                # Receive message
                message_text = await websocket.receive_text()
                await manager.handle_message(connection, message_text)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break
        
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Disconnect
        if 'connection' in locals():
            await manager.disconnect(connection.connection_id, "Connection closed") 
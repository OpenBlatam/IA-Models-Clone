#!/usr/bin/env python3
"""
Real-Time Streaming System

Advanced real-time streaming with:
- WebSocket connections
- Server-Sent Events (SSE)
- Real-time data broadcasting
- Stream management and routing
- Connection pooling and load balancing
- Stream analytics and monitoring
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Set
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import websockets
from websockets.server import WebSocketServerProtocol
from fastapi import WebSocket, WebSocketDisconnect
import sse_starlette

logger = structlog.get_logger("real_time_streaming")

# =============================================================================
# STREAMING MODELS
# =============================================================================

class StreamType(Enum):
    """Stream types."""
    WEBSOCKET = "websocket"
    SERVER_SENT_EVENTS = "server_sent_events"
    LONG_POLLING = "long_polling"
    PUSH_NOTIFICATION = "push_notification"

class StreamStatus(Enum):
    """Stream status."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class MessageType(Enum):
    """Message types for streaming."""
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    NOTIFICATION = "notification"
    CONTROL = "control"

@dataclass
class StreamMessage:
    """Stream message structure."""
    message_id: str
    stream_id: str
    message_type: MessageType
    data: Any
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "stream_id": self.stream_id,
            "message_type": self.message_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

@dataclass
class StreamConnection:
    """Stream connection information."""
    connection_id: str
    stream_id: str
    stream_type: StreamType
    user_id: Optional[str]
    session_id: Optional[str]
    websocket: Optional[WebSocket]
    sse_connection: Optional[Any]
    status: StreamStatus
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.connection_id:
            self.connection_id = str(uuid.uuid4())
        if not self.connected_at:
            self.connected_at = datetime.utcnow()
        if not self.last_activity:
            self.last_activity = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connection_id": self.connection_id,
            "stream_id": self.stream_id,
            "stream_type": self.stream_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class StreamConfig:
    """Stream configuration."""
    stream_id: str
    name: str
    description: str
    stream_type: StreamType
    max_connections: int = 1000
    heartbeat_interval: int = 30
    message_ttl: int = 3600
    buffer_size: int = 1000
    enable_authentication: bool = True
    enable_rate_limiting: bool = True
    rate_limit: int = 100  # messages per minute
    allowed_origins: List[str] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stream_id": self.stream_id,
            "name": self.name,
            "description": self.description,
            "stream_type": self.stream_type.value,
            "max_connections": self.max_connections,
            "heartbeat_interval": self.heartbeat_interval,
            "message_ttl": self.message_ttl,
            "buffer_size": self.buffer_size,
            "enable_authentication": self.enable_authentication,
            "enable_rate_limiting": self.enable_rate_limiting,
            "rate_limit": self.rate_limit,
            "allowed_origins": self.allowed_origins
        }

# =============================================================================
# STREAM MANAGER
# =============================================================================

class StreamManager:
    """Manager for real-time streams."""
    
    def __init__(self):
        self.streams: Dict[str, StreamConfig] = {}
        self.connections: Dict[str, StreamConnection] = {}
        self.stream_connections: Dict[str, Set[str]] = defaultdict(set)
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.message_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics
        self.stats = {
            'total_streams': 0,
            'active_streams': 0,
            'total_connections': 0,
            'active_connections': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0
        }
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start the stream manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Stream manager started")
    
    async def stop(self) -> None:
        """Stop the stream manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self._close_connection(connection)
        
        logger.info("Stream manager stopped")
    
    def create_stream(self, config: StreamConfig) -> str:
        """Create a new stream."""
        self.streams[config.stream_id] = config
        self.stats['total_streams'] += 1
        self.stats['active_streams'] += 1
        
        logger.info(
            "Stream created",
            stream_id=config.stream_id,
            name=config.name,
            type=config.stream_type.value
        )
        
        return config.stream_id
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream."""
        if stream_id not in self.streams:
            return False
        
        # Close all connections for this stream
        connection_ids = list(self.stream_connections[stream_id])
        for connection_id in connection_ids:
            connection = self.connections.get(connection_id)
            if connection:
                asyncio.create_task(self._close_connection(connection))
        
        # Remove stream
        del self.streams[stream_id]
        del self.stream_connections[stream_id]
        del self.message_buffers[stream_id]
        
        self.stats['total_streams'] -= 1
        self.stats['active_streams'] -= 1
        
        logger.info("Stream removed", stream_id=stream_id)
        return True
    
    async def connect_websocket(self, websocket: WebSocket, stream_id: str, user_id: Optional[str] = None) -> str:
        """Connect WebSocket to stream."""
        # Check if stream exists
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream_config = self.streams[stream_id]
        
        # Check connection limit
        if len(self.stream_connections[stream_id]) >= stream_config.max_connections:
            raise RuntimeError(f"Stream {stream_id} connection limit reached")
        
        # Accept WebSocket connection
        await websocket.accept()
        
        # Create connection
        connection = StreamConnection(
            connection_id=str(uuid.uuid4()),
            stream_id=stream_id,
            stream_type=StreamType.WEBSOCKET,
            user_id=user_id,
            websocket=websocket,
            status=StreamStatus.CONNECTED,
            metadata={}
        )
        
        # Register connection
        self._register_connection(connection)
        
        logger.info(
            "WebSocket connected",
            connection_id=connection.connection_id,
            stream_id=stream_id,
            user_id=user_id
        )
        
        return connection.connection_id
    
    async def connect_sse(self, stream_id: str, user_id: Optional[str] = None) -> Any:
        """Connect Server-Sent Events to stream."""
        # Check if stream exists
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream_config = self.streams[stream_id]
        
        # Check connection limit
        if len(self.stream_connections[stream_id]) >= stream_config.max_connections:
            raise RuntimeError(f"Stream {stream_id} connection limit reached")
        
        # Create SSE connection
        sse_connection = sse_starlette.EventSourceResponse(
            self._sse_generator(stream_id, user_id)
        )
        
        # Create connection
        connection = StreamConnection(
            connection_id=str(uuid.uuid4()),
            stream_id=stream_id,
            stream_type=StreamType.SERVER_SENT_EVENTS,
            user_id=user_id,
            sse_connection=sse_connection,
            status=StreamStatus.CONNECTED,
            metadata={}
        )
        
        # Register connection
        self._register_connection(connection)
        
        logger.info(
            "SSE connected",
            connection_id=connection.connection_id,
            stream_id=stream_id,
            user_id=user_id
        )
        
        return sse_connection
    
    async def disconnect(self, connection_id: str) -> bool:
        """Disconnect a connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        await self._close_connection(connection)
        return True
    
    async def send_message(self, stream_id: str, message: StreamMessage) -> int:
        """Send message to stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        # Add to message buffer
        self.message_buffers[stream_id].append(message)
        
        # Send to all connections
        sent_count = 0
        connection_ids = list(self.stream_connections[stream_id])
        
        for connection_id in connection_ids:
            connection = self.connections.get(connection_id)
            if connection and connection.status == StreamStatus.CONNECTED:
                try:
                    if connection.stream_type == StreamType.WEBSOCKET:
                        await self._send_websocket_message(connection, message)
                    elif connection.stream_type == StreamType.SERVER_SENT_EVENTS:
                        await self._send_sse_message(connection, message)
                    
                    sent_count += 1
                    connection.last_activity = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(
                        "Failed to send message",
                        connection_id=connection_id,
                        stream_id=stream_id,
                        error=str(e)
                    )
                    
                    # Mark connection as error
                    connection.status = StreamStatus.ERROR
        
        # Update statistics
        self.stats['total_messages_sent'] += 1
        self.stats['total_bytes_sent'] += len(message.to_json().encode())
        
        logger.debug(
            "Message sent",
            stream_id=stream_id,
            message_id=message.message_id,
            sent_count=sent_count,
            total_connections=len(connection_ids)
        )
        
        return sent_count
    
    async def send_to_user(self, user_id: str, message: StreamMessage) -> int:
        """Send message to specific user."""
        sent_count = 0
        user_connection_ids = list(self.user_connections[user_id])
        
        for connection_id in user_connection_ids:
            connection = self.connections.get(connection_id)
            if connection and connection.status == StreamStatus.CONNECTED:
                try:
                    if connection.stream_type == StreamType.WEBSOCKET:
                        await self._send_websocket_message(connection, message)
                    elif connection.stream_type == StreamType.SERVER_SENT_EVENTS:
                        await self._send_sse_message(connection, message)
                    
                    sent_count += 1
                    connection.last_activity = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(
                        "Failed to send user message",
                        connection_id=connection_id,
                        user_id=user_id,
                        error=str(e)
                    )
        
        return sent_count
    
    async def broadcast_message(self, message: StreamMessage) -> int:
        """Broadcast message to all streams."""
        total_sent = 0
        
        for stream_id in self.streams:
            sent_count = await self.send_message(stream_id, message)
            total_sent += sent_count
        
        return total_sent
    
    def _register_connection(self, connection: StreamConnection) -> None:
        """Register a connection."""
        self.connections[connection.connection_id] = connection
        self.stream_connections[connection.stream_id].add(connection.connection_id)
        
        if connection.user_id:
            self.user_connections[connection.user_id].add(connection.connection_id)
        
        self.stats['total_connections'] += 1
        self.stats['active_connections'] += 1
    
    def _unregister_connection(self, connection: StreamConnection) -> None:
        """Unregister a connection."""
        if connection.connection_id in self.connections:
            del self.connections[connection.connection_id]
        
        self.stream_connections[connection.stream_id].discard(connection.connection_id)
        
        if connection.user_id:
            self.user_connections[connection.user_id].discard(connection.connection_id)
        
        self.stats['active_connections'] -= 1
    
    async def _close_connection(self, connection: StreamConnection) -> None:
        """Close a connection."""
        try:
            if connection.stream_type == StreamType.WEBSOCKET and connection.websocket:
                await connection.websocket.close()
            elif connection.stream_type == StreamType.SERVER_SENT_EVENTS and connection.sse_connection:
                # SSE connections are handled by the generator
                pass
            
            connection.status = StreamStatus.DISCONNECTED
            
        except Exception as e:
            logger.error("Error closing connection", connection_id=connection.connection_id, error=str(e))
        
        finally:
            self._unregister_connection(connection)
    
    async def _send_websocket_message(self, connection: StreamConnection, message: StreamMessage) -> None:
        """Send message via WebSocket."""
        if connection.websocket:
            await connection.websocket.send_text(message.to_json())
    
    async def _send_sse_message(self, connection: StreamConnection, message: StreamMessage) -> None:
        """Send message via SSE."""
        # SSE messages are handled by the generator
        pass
    
    async def _sse_generator(self, stream_id: str, user_id: Optional[str]):
        """SSE message generator."""
        connection_id = None
        
        try:
            # Find the connection
            for conn_id, connection in self.connections.items():
                if (connection.stream_id == stream_id and 
                    connection.user_id == user_id and 
                    connection.stream_type == StreamType.SERVER_SENT_EVENTS):
                    connection_id = conn_id
                    break
            
            if not connection_id:
                return
            
            connection = self.connections[connection_id]
            connection.status = StreamStatus.CONNECTED
            
            # Send initial heartbeat
            heartbeat_message = StreamMessage(
                stream_id=stream_id,
                message_type=MessageType.HEARTBEAT,
                data={"status": "connected"},
                user_id=user_id
            )
            
            yield {
                "event": "heartbeat",
                "data": heartbeat_message.to_json()
            }
            
            # Send buffered messages
            for message in self.message_buffers[stream_id]:
                yield {
                    "event": message.message_type.value,
                    "data": message.to_json()
                }
            
            # Keep connection alive and send new messages
            while connection.status == StreamStatus.CONNECTED:
                try:
                    # Check for new messages
                    if self.message_buffers[stream_id]:
                        latest_message = self.message_buffers[stream_id][-1]
                        if latest_message.timestamp > connection.last_activity:
                            yield {
                                "event": latest_message.message_type.value,
                                "data": latest_message.to_json()
                            }
                            connection.last_activity = datetime.utcnow()
                    
                    # Send heartbeat
                    heartbeat_message = StreamMessage(
                        stream_id=stream_id,
                        message_type=MessageType.HEARTBEAT,
                        data={"timestamp": datetime.utcnow().isoformat()},
                        user_id=user_id
                    )
                    
                    yield {
                        "event": "heartbeat",
                        "data": heartbeat_message.to_json()
                    }
                    
                    await asyncio.sleep(30)  # Heartbeat interval
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("SSE generator error", error=str(e))
                    break
        
        except Exception as e:
            logger.error("SSE connection error", stream_id=stream_id, error=str(e))
        
        finally:
            if connection_id:
                connection = self.connections.get(connection_id)
                if connection:
                    connection.status = StreamStatus.DISCONNECTED
                    self._unregister_connection(connection)
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop for all connections."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for connection in list(self.connections.values()):
                    if connection.status != StreamStatus.CONNECTED:
                        continue
                    
                    # Check if connection is alive
                    time_since_activity = (current_time - connection.last_activity).total_seconds()
                    
                    if time_since_activity > 60:  # 1 minute timeout
                        logger.warning(
                            "Connection timeout",
                            connection_id=connection.connection_id,
                            time_since_activity=time_since_activity
                        )
                        await self._close_connection(connection)
                        continue
                    
                    # Send heartbeat
                    heartbeat_message = StreamMessage(
                        stream_id=connection.stream_id,
                        message_type=MessageType.HEARTBEAT,
                        data={"timestamp": current_time.isoformat()},
                        user_id=connection.user_id
                    )
                    
                    try:
                        if connection.stream_type == StreamType.WEBSOCKET:
                            await self._send_websocket_message(connection, heartbeat_message)
                    except Exception as e:
                        logger.error("Heartbeat failed", connection_id=connection.connection_id, error=str(e))
                        await self._close_connection(connection)
                
                await asyncio.sleep(30)  # Heartbeat interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired messages and connections."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired messages
                for stream_id, buffer in self.message_buffers.items():
                    stream_config = self.streams.get(stream_id)
                    if not stream_config:
                        continue
                    
                    ttl_seconds = stream_config.message_ttl
                    expired_messages = [
                        msg for msg in buffer
                        if (current_time - msg.timestamp).total_seconds() > ttl_seconds
                    ]
                    
                    for msg in expired_messages:
                        buffer.remove(msg)
                
                # Clean up disconnected connections
                disconnected_connections = [
                    conn for conn in self.connections.values()
                    if conn.status == StreamStatus.DISCONNECTED
                ]
                
                for connection in disconnected_connections:
                    self._unregister_connection(connection)
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(300)
    
    def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """Get stream statistics."""
        if stream_id not in self.streams:
            return {}
        
        stream_config = self.streams[stream_id]
        connection_count = len(self.stream_connections[stream_id])
        message_count = len(self.message_buffers[stream_id])
        
        return {
            'stream_id': stream_id,
            'name': stream_config.name,
            'type': stream_config.stream_type.value,
            'connection_count': connection_count,
            'max_connections': stream_config.max_connections,
            'message_count': message_count,
            'buffer_size': stream_config.buffer_size,
            'created_at': stream_config.to_dict()
        }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'streams': {
                stream_id: self.get_stream_stats(stream_id)
                for stream_id in self.streams
            },
            'connections_by_stream': {
                stream_id: len(connections)
                for stream_id, connections in self.stream_connections.items()
            },
            'connections_by_user': {
                user_id: len(connections)
                for user_id, connections in self.user_connections.items()
            }
        }

# =============================================================================
# GLOBAL STREAMING INSTANCES
# =============================================================================

# Global stream manager
stream_manager = StreamManager()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StreamType',
    'StreamStatus',
    'MessageType',
    'StreamMessage',
    'StreamConnection',
    'StreamConfig',
    'StreamManager',
    'stream_manager'
]






























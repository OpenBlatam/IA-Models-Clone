"""
WebSocket Manager
=================

WebSocket connection management and message broadcasting.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from .types import (
    WebSocketMessage, MessageType, ConnectionInfo, 
    BroadcastMessage, WebSocketError, ConnectionError
)

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        self.topic_subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()
            
            connection_id = str(uuid.uuid4())
            
            async with self._lock:
                self.active_connections[connection_id] = websocket
                
                connection_info = ConnectionInfo(
                    connection_id=connection_id,
                    user_id=user_id,
                    session_id=session_id
                )
                self.connection_info[connection_id] = connection_info
                
                # Track user connections
                if user_id:
                    if user_id not in self.user_connections:
                        self.user_connections[user_id] = set()
                    self.user_connections[user_id].add(connection_id)
                
                # Track session connections
                if session_id:
                    if session_id not in self.session_connections:
                        self.session_connections[session_id] = set()
                    self.session_connections[session_id].add(connection_id)
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                type=MessageType.CONNECT,
                data={
                    "connection_id": connection_id,
                    "message": "Connected to Business Agents System",
                    "server_time": datetime.now().isoformat()
                },
                message_id=str(uuid.uuid4())
            )
            
            await self.send_to_connection(connection_id, welcome_message)
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {str(e)}")
            raise ConnectionError(f"Failed to establish connection: {str(e)}")
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection."""
        try:
            async with self._lock:
                if connection_id in self.active_connections:
                    websocket = self.active_connections[connection_id]
                    connection_info = self.connection_info.get(connection_id)
                    
                    # Close the WebSocket
                    try:
                        await websocket.close()
                    except Exception:
                        pass  # Connection might already be closed
                    
                    # Clean up tracking
                    del self.active_connections[connection_id]
                    
                    if connection_info:
                        # Remove from user connections
                        if connection_info.user_id and connection_info.user_id in self.user_connections:
                            self.user_connections[connection_info.user_id].discard(connection_id)
                            if not self.user_connections[connection_info.user_id]:
                                del self.user_connections[connection_info.user_id]
                        
                        # Remove from session connections
                        if connection_info.session_id and connection_info.session_id in self.session_connections:
                            self.session_connections[connection_info.session_id].discard(connection_id)
                            if not self.session_connections[connection_info.session_id]:
                                del self.session_connections[connection_info.session_id]
                        
                        # Remove from topic subscriptions
                        for topic in connection_info.subscriptions:
                            if topic in self.topic_subscriptions:
                                self.topic_subscriptions[topic].discard(connection_id)
                                if not self.topic_subscriptions[topic]:
                                    del self.topic_subscriptions[topic]
                        
                        del self.connection_info[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}")
            
        except Exception as e:
            logger.error(f"Failed to disconnect WebSocket connection {connection_id}: {str(e)}")
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send a message to a specific connection."""
        try:
            if connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message.to_json())
            
            # Update last activity
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_activity = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {str(e)}")
            # Remove failed connection
            await self.disconnect(connection_id)
            return False
    
    async def broadcast(self, broadcast_message: BroadcastMessage) -> int:
        """Broadcast a message to multiple connections."""
        sent_count = 0
        
        try:
            connections_to_send = []
            
            # Determine target connections
            if broadcast_message.target_connections:
                connections_to_send = broadcast_message.target_connections
            elif broadcast_message.target_users:
                for user_id in broadcast_message.target_users:
                    if user_id in self.user_connections:
                        connections_to_send.extend(self.user_connections[user_id])
            elif broadcast_message.target_sessions:
                for session_id in broadcast_message.target_sessions:
                    if session_id in self.session_connections:
                        connections_to_send.extend(self.session_connections[session_id])
            elif broadcast_message.topic:
                if broadcast_message.topic in self.topic_subscriptions:
                    connections_to_send = list(self.topic_subscriptions[broadcast_message.topic])
            else:
                # Broadcast to all connections
                connections_to_send = list(self.active_connections.keys())
            
            # Send to each connection
            for connection_id in connections_to_send:
                if connection_id in self.connection_info:
                    connection_info = self.connection_info[connection_id]
                    
                    if broadcast_message.should_send_to_connection(connection_info):
                        success = await self.send_to_connection(connection_id, broadcast_message.message)
                        if success:
                            sent_count += 1
            
            logger.debug(f"Broadcasted message to {sent_count} connections")
            return sent_count
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {str(e)}")
            return sent_count
    
    async def subscribe_to_topic(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic."""
        try:
            if connection_id not in self.connection_info:
                return False
            
            async with self._lock:
                connection_info = self.connection_info[connection_id]
                connection_info.add_subscription(topic)
                
                if topic not in self.topic_subscriptions:
                    self.topic_subscriptions[topic] = set()
                self.topic_subscriptions[topic].add(connection_id)
            
            logger.info(f"Connection {connection_id} subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe connection {connection_id} to topic {topic}: {str(e)}")
            return False
    
    async def unsubscribe_from_topic(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic."""
        try:
            if connection_id not in self.connection_info:
                return False
            
            async with self._lock:
                connection_info = self.connection_info[connection_id]
                connection_info.remove_subscription(topic)
                
                if topic in self.topic_subscriptions:
                    self.topic_subscriptions[topic].discard(connection_id)
                    if not self.topic_subscriptions[topic]:
                        del self.topic_subscriptions[topic]
            
            logger.info(f"Connection {connection_id} unsubscribed from topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe connection {connection_id} from topic {topic}: {str(e)}")
            return False
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection information."""
        return self.connection_info.get(connection_id)
    
    def get_active_connections_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_connections_by_user(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user."""
        return list(self.user_connections.get(user_id, set()))
    
    def get_connections_by_session(self, session_id: str) -> List[str]:
        """Get all connection IDs for a session."""
        return list(self.session_connections.get(session_id, set()))
    
    def get_subscribers_for_topic(self, topic: str) -> List[str]:
        """Get all connection IDs subscribed to a topic."""
        return list(self.topic_subscriptions.get(topic, set()))
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 30):
        """Clean up stale connections."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            stale_connections = []
            
            async with self._lock:
                for connection_id, connection_info in self.connection_info.items():
                    if connection_info.last_activity < cutoff_time:
                        stale_connections.append(connection_id)
            
            for connection_id in stale_connections:
                await self.disconnect(connection_id)
            
            if stale_connections:
                logger.info(f"Cleaned up {len(stale_connections)} stale connections")
            
        except Exception as e:
            logger.error(f"Failed to cleanup stale connections: {str(e)}")

class WebSocketManager:
    """Main WebSocket manager for the Business Agents System."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.connection_manager = ConnectionManager()
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the WebSocket manager."""
        try:
            # Initialize Redis client for pub/sub
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._running = True
            logger.info("WebSocket manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the WebSocket manager."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection_id in list(self.connection_manager.active_connections.keys()):
            await self.connection_manager.disconnect(connection_id)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("WebSocket manager shutdown complete")
    
    async def handle_connection(self, websocket: WebSocket, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Handle a new WebSocket connection."""
        connection_id = None
        
        try:
            connection_id = await self.connection_manager.connect(websocket, user_id, session_id)
            
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                
                try:
                    message = WebSocketMessage.from_json(data)
                    await self._handle_message(connection_id, message)
                    
                except json.JSONDecodeError:
                    error_message = WebSocketMessage(
                        type=MessageType.CUSTOM_EVENT,
                        data={"error": "Invalid JSON format"},
                        message_id=str(uuid.uuid4())
                    )
                    await self.connection_manager.send_to_connection(connection_id, error_message)
                
                except Exception as e:
                    logger.error(f"Error handling message from {connection_id}: {str(e)}")
                    error_message = WebSocketMessage(
                        type=MessageType.CUSTOM_EVENT,
                        data={"error": "Message processing failed"},
                        message_id=str(uuid.uuid4())
                    )
                    await self.connection_manager.send_to_connection(connection_id, error_message)
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
        finally:
            if connection_id:
                await self.connection_manager.disconnect(connection_id)
    
    async def _handle_message(self, connection_id: str, message: WebSocketMessage):
        """Handle incoming WebSocket message."""
        try:
            # Update connection activity
            if connection_id in self.connection_manager.connection_info:
                self.connection_manager.connection_info[connection_id].last_activity = datetime.now()
            
            # Handle different message types
            if message.type == MessageType.PING:
                # Respond with pong
                pong_message = WebSocketMessage(
                    type=MessageType.PONG,
                    data={"timestamp": datetime.now().isoformat()},
                    message_id=str(uuid.uuid4())
                )
                await self.connection_manager.send_to_connection(connection_id, pong_message)
            
            elif message.type == MessageType.CUSTOM_EVENT:
                # Handle custom events (like subscriptions)
                event_type = message.data.get("event_type")
                
                if event_type == "subscribe":
                    topic = message.data.get("topic")
                    if topic:
                        await self.connection_manager.subscribe_to_topic(connection_id, topic)
                        
                        response = WebSocketMessage(
                            type=MessageType.CUSTOM_EVENT,
                            data={
                                "event_type": "subscription_confirmed",
                                "topic": topic,
                                "message": f"Subscribed to {topic}"
                            },
                            message_id=str(uuid.uuid4())
                        )
                        await self.connection_manager.send_to_connection(connection_id, response)
                
                elif event_type == "unsubscribe":
                    topic = message.data.get("topic")
                    if topic:
                        await self.connection_manager.unsubscribe_from_topic(connection_id, topic)
                        
                        response = WebSocketMessage(
                            type=MessageType.CUSTOM_EVENT,
                            data={
                                "event_type": "unsubscription_confirmed",
                                "topic": topic,
                                "message": f"Unsubscribed from {topic}"
                            },
                            message_id=str(uuid.uuid4())
                        )
                        await self.connection_manager.send_to_connection(connection_id, response)
            
            # Call registered message handlers
            if message.type in self.message_handlers:
                for handler in self.message_handlers[message.type]:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to handle message: {str(e)}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def broadcast_agent_execution_update(
        self, 
        agent_id: str, 
        capability_name: str, 
        status: str, 
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Broadcast agent execution update."""
        message = WebSocketMessage(
            type=MessageType.AGENT_EXECUTION_PROGRESS,
            data={
                "agent_id": agent_id,
                "capability_name": capability_name,
                "status": status,
                "progress": progress,
                "result": result,
                "error": error,
                "timestamp": datetime.now().isoformat()
            },
            message_id=str(uuid.uuid4())
        )
        
        broadcast_message = BroadcastMessage(
            message=message,
            topic=f"agent_execution:{agent_id}"
        )
        
        await self.connection_manager.broadcast(broadcast_message)
    
    async def broadcast_workflow_execution_update(
        self,
        workflow_id: str,
        status: str,
        current_step: Optional[str] = None,
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Broadcast workflow execution update."""
        message = WebSocketMessage(
            type=MessageType.WORKFLOW_EXECUTION_PROGRESS,
            data={
                "workflow_id": workflow_id,
                "status": status,
                "current_step": current_step,
                "progress": progress,
                "result": result,
                "error": error,
                "timestamp": datetime.now().isoformat()
            },
            message_id=str(uuid.uuid4())
        )
        
        broadcast_message = BroadcastMessage(
            message=message,
            topic=f"workflow_execution:{workflow_id}"
        )
        
        await self.connection_manager.broadcast(broadcast_message)
    
    async def broadcast_system_alert(
        self,
        alert_type: str,
        severity: str,
        message_text: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Broadcast system alert."""
        message = WebSocketMessage(
            type=MessageType.SYSTEM_ALERT,
            data={
                "alert_type": alert_type,
                "severity": severity,
                "message": message_text,
                "details": details,
                "timestamp": datetime.now().isoformat()
            },
            message_id=str(uuid.uuid4())
        )
        
        broadcast_message = BroadcastMessage(
            message=message,
            topic="system_alerts"
        )
        
        await self.connection_manager.broadcast(broadcast_message)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await self.connection_manager.cleanup_stale_connections()
                await asyncio.sleep(300)  # Run every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

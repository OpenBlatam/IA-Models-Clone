"""
WebSocket Service - Advanced Implementation
==========================================

Advanced WebSocket service with real-time communication and event broadcasting.
"""

from __future__ import annotations
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class WebSocketEventType(str, Enum):
    """WebSocket event type enumeration"""
    WORKFLOW_UPDATE = "workflow_update"
    WORKFLOW_EXECUTION = "workflow_execution"
    NODE_UPDATE = "node_update"
    NODE_EXECUTION = "node_execution"
    USER_ACTIVITY = "user_activity"
    SYSTEM_NOTIFICATION = "system_notification"
    AI_PROCESSING = "ai_processing"
    CACHE_UPDATE = "cache_update"
    SECURITY_EVENT = "security_event"
    AUDIT_EVENT = "audit_event"
    ANALYTICS_UPDATE = "analytics_update"
    NOTIFICATION_SENT = "notification_sent"


class WebSocketConnectionManager:
    """Advanced WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.room_connections: Dict[str, Set[str]] = {}
        self.user_connections: Dict[int, Set[str]] = {}
        self.event_handlers: Dict[WebSocketEventType, List[Callable]] = {}
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "events_broadcasted": 0
        }
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """Accept WebSocket connection"""
        try:
            await websocket.accept()
            
            self.active_connections[connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "connected_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Track user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            
            # Send welcome message
            await self.send_personal_message({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to Document Workflow Chain WebSocket"
            }, connection_id)
            
            logger.info(f"WebSocket connected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise
    
    def disconnect(self, connection_id: str):
        """Disconnect WebSocket connection"""
        try:
            if connection_id in self.active_connections:
                # Remove from active connections
                del self.active_connections[connection_id]
                
                # Remove from user connections
                metadata = self.connection_metadata.get(connection_id, {})
                user_id = metadata.get("user_id")
                if user_id and user_id in self.user_connections:
                    self.user_connections[user_id].discard(connection_id)
                    if not self.user_connections[user_id]:
                        del self.user_connections[user_id]
                
                # Remove from rooms
                for room_id, connections in self.room_connections.items():
                    connections.discard(connection_id)
                
                # Remove metadata
                if connection_id in self.connection_metadata:
                    del self.connection_metadata[connection_id]
                
                self.stats["active_connections"] -= 1
                
                logger.info(f"WebSocket disconnected: {connection_id}")
        
        except Exception as e:
            logger.error(f"Failed to disconnect WebSocket: {e}")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send message to specific connection"""
        try:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                    self.stats["messages_sent"] += 1
                    
                    # Update last activity
                    if connection_id in self.connection_metadata:
                        self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow().isoformat()
        
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(connection_id)
    
    async def send_to_user(self, message: Dict[str, Any], user_id: int):
        """Send message to all connections of a user"""
        try:
            if user_id in self.user_connections:
                for connection_id in self.user_connections[user_id].copy():
                    await self.send_personal_message(message, connection_id)
        
        except Exception as e:
            logger.error(f"Failed to send message to user: {e}")
    
    async def send_to_room(self, message: Dict[str, Any], room_id: str):
        """Send message to all connections in a room"""
        try:
            if room_id in self.room_connections:
                for connection_id in self.room_connections[room_id].copy():
                    await self.send_personal_message(message, connection_id)
        
        except Exception as e:
            logger.error(f"Failed to send message to room: {e}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        try:
            for connection_id in list(self.active_connections.keys()):
                await self.send_personal_message(message, connection_id)
        
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    async def join_room(self, connection_id: str, room_id: str):
        """Join connection to a room"""
        try:
            if room_id not in self.room_connections:
                self.room_connections[room_id] = set()
            
            self.room_connections[room_id].add(connection_id)
            
            # Notify room members
            await self.send_to_room({
                "type": "user_joined_room",
                "room_id": room_id,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }, room_id)
            
            logger.info(f"Connection {connection_id} joined room {room_id}")
        
        except Exception as e:
            logger.error(f"Failed to join room: {e}")
    
    async def leave_room(self, connection_id: str, room_id: str):
        """Leave connection from a room"""
        try:
            if room_id in self.room_connections:
                self.room_connections[room_id].discard(connection_id)
                
                # Notify room members
                await self.send_to_room({
                    "type": "user_left_room",
                    "room_id": room_id,
                    "connection_id": connection_id,
                    "timestamp": datetime.utcnow().isoformat()
                }, room_id)
                
                logger.info(f"Connection {connection_id} left room {room_id}")
        
        except Exception as e:
            logger.error(f"Failed to leave room: {e}")
    
    async def broadcast_event(self, event_type: WebSocketEventType, data: Dict[str, Any], target: Optional[str] = None):
        """Broadcast event to connections"""
        try:
            message = {
                "type": "event",
                "event_type": event_type.value,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if target == "all":
                await self.broadcast(message)
            elif target and target.startswith("user:"):
                user_id = int(target.split(":")[1])
                await self.send_to_user(message, user_id)
            elif target and target.startswith("room:"):
                room_id = target.split(":")[1]
                await self.send_to_room(message, room_id)
            else:
                await self.broadcast(message)
            
            self.stats["events_broadcasted"] += 1
            
            # Call event handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(event_type, data)
                    except Exception as e:
                        logger.error(f"Event handler failed: {e}")
        
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
    
    def register_event_handler(self, event_type: WebSocketEventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information"""
        return self.connection_metadata.get(connection_id)
    
    def get_user_connections(self, user_id: int) -> List[str]:
        """Get all connections for a user"""
        return list(self.user_connections.get(user_id, set()))
    
    def get_room_connections(self, room_id: str) -> List[str]:
        """Get all connections in a room"""
        return list(self.room_connections.get(room_id, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics"""
        return {
            "total_connections": self.stats["total_connections"],
            "active_connections": self.stats["active_connections"],
            "messages_sent": self.stats["messages_sent"],
            "messages_received": self.stats["messages_received"],
            "events_broadcasted": self.stats["events_broadcasted"],
            "rooms_count": len(self.room_connections),
            "users_connected": len(self.user_connections),
            "timestamp": datetime.utcnow().isoformat()
        }


class WebSocketService:
    """Advanced WebSocket service with real-time communication"""
    
    def __init__(self):
        self.manager = WebSocketConnectionManager()
        self.workflow_rooms = {}  # workflow_id -> room_id
        self.user_rooms = {}      # user_id -> room_id
        self.system_room = "system"
        
        # Register default event handlers
        self._register_default_handlers()
    
    async def handle_connection(self, websocket: WebSocket, connection_id: str, user_id: Optional[int] = None):
        """Handle WebSocket connection"""
        try:
            await self.manager.connect(websocket, connection_id, user_id)
            
            # Join system room
            await self.manager.join_room(connection_id, self.system_room)
            
            # Join user room if user_id provided
            if user_id:
                user_room = f"user_{user_id}"
                await self.manager.join_room(connection_id, user_room)
                self.user_rooms[user_id] = user_room
            
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_message(connection_id, message)
                    self.manager.stats["messages_received"] += 1
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self.manager.send_personal_message({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self.manager.send_personal_message({
                        "type": "error",
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.manager.disconnect(connection_id)
    
    async def _handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get("type")
            
            if message_type == "join_workflow":
                workflow_id = message.get("workflow_id")
                if workflow_id:
                    await self._join_workflow_room(connection_id, workflow_id)
            
            elif message_type == "leave_workflow":
                workflow_id = message.get("workflow_id")
                if workflow_id:
                    await self._leave_workflow_room(connection_id, workflow_id)
            
            elif message_type == "ping":
                await self.manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            elif message_type == "get_stats":
                stats = self.manager.get_stats()
                await self.manager.send_personal_message({
                    "type": "stats",
                    "data": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            else:
                await self.manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
        
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
    
    async def _join_workflow_room(self, connection_id: str, workflow_id: int):
        """Join workflow room"""
        try:
            room_id = f"workflow_{workflow_id}"
            await self.manager.join_room(connection_id, room_id)
            self.workflow_rooms[workflow_id] = room_id
            
            await self.manager.send_personal_message({
                "type": "joined_workflow",
                "workflow_id": workflow_id,
                "room_id": room_id,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
        
        except Exception as e:
            logger.error(f"Failed to join workflow room: {e}")
    
    async def _leave_workflow_room(self, connection_id: str, workflow_id: int):
        """Leave workflow room"""
        try:
            room_id = f"workflow_{workflow_id}"
            await self.manager.leave_room(connection_id, room_id)
            
            await self.manager.send_personal_message({
                "type": "left_workflow",
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
        
        except Exception as e:
            logger.error(f"Failed to leave workflow room: {e}")
    
    async def broadcast_workflow_update(self, workflow_id: int, update_data: Dict[str, Any]):
        """Broadcast workflow update"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.WORKFLOW_UPDATE,
                {
                    "workflow_id": workflow_id,
                    "update": update_data
                },
                f"room:workflow_{workflow_id}"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast workflow update: {e}")
    
    async def broadcast_workflow_execution(self, workflow_id: int, execution_data: Dict[str, Any]):
        """Broadcast workflow execution"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.WORKFLOW_EXECUTION,
                {
                    "workflow_id": workflow_id,
                    "execution": execution_data
                },
                f"room:workflow_{workflow_id}"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast workflow execution: {e}")
    
    async def broadcast_node_update(self, workflow_id: int, node_id: int, update_data: Dict[str, Any]):
        """Broadcast node update"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.NODE_UPDATE,
                {
                    "workflow_id": workflow_id,
                    "node_id": node_id,
                    "update": update_data
                },
                f"room:workflow_{workflow_id}"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast node update: {e}")
    
    async def broadcast_ai_processing(self, user_id: int, processing_data: Dict[str, Any]):
        """Broadcast AI processing update"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.AI_PROCESSING,
                {
                    "user_id": user_id,
                    "processing": processing_data
                },
                f"user:{user_id}"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast AI processing: {e}")
    
    async def broadcast_system_notification(self, notification_data: Dict[str, Any]):
        """Broadcast system notification"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.SYSTEM_NOTIFICATION,
                notification_data,
                "all"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast system notification: {e}")
    
    async def broadcast_security_event(self, security_data: Dict[str, Any]):
        """Broadcast security event"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.SECURITY_EVENT,
                security_data,
                "all"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast security event: {e}")
    
    async def broadcast_analytics_update(self, analytics_data: Dict[str, Any]):
        """Broadcast analytics update"""
        try:
            await self.manager.broadcast_event(
                WebSocketEventType.ANALYTICS_UPDATE,
                analytics_data,
                "all"
            )
        
        except Exception as e:
            logger.error(f"Failed to broadcast analytics update: {e}")
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        # Register handlers for different event types
        self.manager.register_event_handler(
            WebSocketEventType.WORKFLOW_UPDATE,
            self._handle_workflow_update
        )
        
        self.manager.register_event_handler(
            WebSocketEventType.SECURITY_EVENT,
            self._handle_security_event
        )
    
    async def _handle_workflow_update(self, event_type: WebSocketEventType, data: Dict[str, Any]):
        """Handle workflow update event"""
        try:
            logger.info(f"Workflow update event: {data}")
        except Exception as e:
            logger.error(f"Failed to handle workflow update: {e}")
    
    async def _handle_security_event(self, event_type: WebSocketEventType, data: Dict[str, Any]):
        """Handle security event"""
        try:
            logger.warning(f"Security event broadcasted: {data}")
        except Exception as e:
            logger.error(f"Failed to handle security event: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return self.manager.get_stats()
    
    def get_active_connections(self) -> int:
        """Get number of active connections"""
        return self.manager.stats["active_connections"]
    
    def get_user_connections(self, user_id: int) -> List[str]:
        """Get user connections"""
        return self.manager.get_user_connections(user_id)


# Global WebSocket service instance
websocket_service = WebSocketService()


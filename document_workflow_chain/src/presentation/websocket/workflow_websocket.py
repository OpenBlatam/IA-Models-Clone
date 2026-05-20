"""
Workflow WebSocket Handler
==========================

Real-time WebSocket communication for workflow updates.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState

from ...shared.container import Container
from ...shared.utils.decorators import log_execution
from ...shared.utils.helpers import DateTimeHelpers, StringHelpers
from ...shared.events.event_bus import EventBus
from ...domain.value_objects.workflow_id import WorkflowId
from ...domain.value_objects.node_id import NodeId


logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.workflow_subscriptions: Dict[str, Set[str]] = {}  # workflow_id -> set of connection_ids
        self.connection_workflows: Dict[str, Set[str]] = {}  # connection_id -> set of workflow_ids
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from workflow subscriptions
        if connection_id in self.connection_workflows:
            for workflow_id in self.connection_workflows[connection_id]:
                if workflow_id in self.workflow_subscriptions:
                    self.workflow_subscriptions[workflow_id].discard(connection_id)
                    if not self.workflow_subscriptions[workflow_id]:
                        del self.workflow_subscriptions[workflow_id]
            del self.connection_workflows[connection_id]
        
        # Remove from user connections
        for user_id, connections in self.user_connections.items():
            connections.discard(connection_id)
            if not connections:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send message to {connection_id}: {e}")
                    self.disconnect(connection_id)
    
    async def send_workflow_update(self, workflow_id: str, message: str):
        """Send message to all connections subscribed to workflow"""
        if workflow_id in self.workflow_subscriptions:
            for connection_id in self.workflow_subscriptions[workflow_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def send_user_message(self, user_id: str, message: str):
        """Send message to all connections of a user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, connection_id)
    
    def subscribe_to_workflow(self, connection_id: str, workflow_id: str):
        """Subscribe connection to workflow updates"""
        if connection_id not in self.connection_workflows:
            self.connection_workflows[connection_id] = set()
        
        self.connection_workflows[connection_id].add(workflow_id)
        
        if workflow_id not in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id] = set()
        
        self.workflow_subscriptions[workflow_id].add(connection_id)
        
        logger.info(f"Connection {connection_id} subscribed to workflow {workflow_id}")
    
    def unsubscribe_from_workflow(self, connection_id: str, workflow_id: str):
        """Unsubscribe connection from workflow updates"""
        if connection_id in self.connection_workflows:
            self.connection_workflows[connection_id].discard(workflow_id)
        
        if workflow_id in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id].discard(connection_id)
            if not self.workflow_subscriptions[workflow_id]:
                del self.workflow_subscriptions[workflow_id]
        
        logger.info(f"Connection {connection_id} unsubscribed from workflow {workflow_id}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "total_users": len(self.user_connections),
            "total_workflow_subscriptions": len(self.workflow_subscriptions),
            "workflows_with_subscribers": len(self.workflow_subscriptions)
        }


# Global connection manager
manager = ConnectionManager()


class WebSocketMessage:
    """WebSocket message handler"""
    
    @staticmethod
    def create_message(message_type: str, data: Dict[str, Any], timestamp: Optional[str] = None) -> str:
        """Create WebSocket message"""
        return json.dumps({
            "type": message_type,
            "data": data,
            "timestamp": timestamp or DateTimeHelpers.now_utc().isoformat()
        })
    
    @staticmethod
    def parse_message(message: str) -> Dict[str, Any]:
        """Parse WebSocket message"""
        try:
            return json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
            return {}
    
    @staticmethod
    def create_workflow_update(workflow_id: str, update_type: str, data: Dict[str, Any]) -> str:
        """Create workflow update message"""
        return WebSocketMessage.create_message(
            "workflow_update",
            {
                "workflow_id": workflow_id,
                "update_type": update_type,
                "update_data": data
            }
        )
    
    @staticmethod
    def create_node_update(workflow_id: str, node_id: str, update_type: str, data: Dict[str, Any]) -> str:
        """Create node update message"""
        return WebSocketMessage.create_message(
            "node_update",
            {
                "workflow_id": workflow_id,
                "node_id": node_id,
                "update_type": update_type,
                "update_data": data
            }
        )
    
    @staticmethod
    def create_system_notification(notification_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Create system notification message"""
        return WebSocketMessage.create_message(
            "system_notification",
            {
                "notification_type": notification_type,
                "message": message,
                "data": data or {}
            }
        )
    
    @staticmethod
    def create_error_message(error_code: str, error_message: str) -> str:
        """Create error message"""
        return WebSocketMessage.create_message(
            "error",
            {
                "error_code": error_code,
                "error_message": error_message
            }
        )


class WebSocketHandler:
    """WebSocket message handler"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.event_bus = Container().get_event_bus()
    
    async def handle_message(self, websocket: WebSocket, connection_id: str, message: str, user_id: Optional[str] = None):
        """Handle incoming WebSocket message"""
        try:
            parsed_message = WebSocketMessage.parse_message(message)
            if not parsed_message:
                await self.send_error(websocket, "INVALID_MESSAGE", "Invalid message format")
                return
            
            message_type = parsed_message.get("type")
            data = parsed_message.get("data", {})
            
            if message_type == "subscribe_workflow":
                await self.handle_subscribe_workflow(connection_id, data)
            elif message_type == "unsubscribe_workflow":
                await self.handle_unsubscribe_workflow(connection_id, data)
            elif message_type == "ping":
                await self.handle_ping(websocket)
            elif message_type == "get_stats":
                await self.handle_get_stats(websocket)
            else:
                await self.send_error(websocket, "UNKNOWN_MESSAGE_TYPE", f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error(websocket, "HANDLER_ERROR", str(e))
    
    async def handle_subscribe_workflow(self, connection_id: str, data: Dict[str, Any]):
        """Handle workflow subscription"""
        workflow_id = data.get("workflow_id")
        if not workflow_id:
            return
        
        # Validate workflow ID format
        try:
            WorkflowId(workflow_id)
            self.connection_manager.subscribe_to_workflow(connection_id, workflow_id)
            
            # Send confirmation
            if connection_id in self.connection_manager.active_connections:
                websocket = self.connection_manager.active_connections[connection_id]
                confirmation = WebSocketMessage.create_message(
                    "subscription_confirmed",
                    {"workflow_id": workflow_id, "status": "subscribed"}
                )
                await websocket.send_text(confirmation)
        
        except ValueError:
            logger.error(f"Invalid workflow ID format: {workflow_id}")
    
    async def handle_unsubscribe_workflow(self, connection_id: str, data: Dict[str, Any]):
        """Handle workflow unsubscription"""
        workflow_id = data.get("workflow_id")
        if not workflow_id:
            return
        
        self.connection_manager.unsubscribe_from_workflow(connection_id, workflow_id)
        
        # Send confirmation
        if connection_id in self.connection_manager.active_connections:
            websocket = self.connection_manager.active_connections[connection_id]
            confirmation = WebSocketMessage.create_message(
                "unsubscription_confirmed",
                {"workflow_id": workflow_id, "status": "unsubscribed"}
            )
            await websocket.send_text(confirmation)
    
    async def handle_ping(self, websocket: WebSocket):
        """Handle ping message"""
        pong = WebSocketMessage.create_message("pong", {"timestamp": DateTimeHelpers.now_utc().isoformat()})
        await websocket.send_text(pong)
    
    async def handle_get_stats(self, websocket: WebSocket):
        """Handle get stats message"""
        stats = self.connection_manager.get_connection_stats()
        stats_message = WebSocketMessage.create_message("connection_stats", stats)
        await websocket.send_text(stats_message)
    
    async def send_error(self, websocket: WebSocket, error_code: str, error_message: str):
        """Send error message"""
        error = WebSocketMessage.create_error_message(error_code, error_message)
        await websocket.send_text(error)


# WebSocket endpoints
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID"),
    connection_id: Optional[str] = Query(None, description="Connection ID")
):
    """WebSocket endpoint for real-time communication"""
    # Generate connection ID if not provided
    if not connection_id:
        connection_id = StringHelpers.generate_random_string(16)
    
    # Create message handler
    handler = WebSocketHandler(manager)
    
    try:
        # Connect
        await manager.connect(websocket, connection_id, user_id)
        
        # Send welcome message
        welcome = WebSocketMessage.create_message(
            "welcome",
            {
                "connection_id": connection_id,
                "user_id": user_id,
                "server_time": DateTimeHelpers.now_utc().isoformat(),
                "available_commands": [
                    "subscribe_workflow",
                    "unsubscribe_workflow",
                    "ping",
                    "get_stats"
                ]
            }
        )
        await websocket.send_text(welcome)
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                await handler.handle_message(websocket, connection_id, data, user_id)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await handler.send_error(websocket, "PROCESSING_ERROR", str(e))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(connection_id)


# Event handlers for real-time updates
class WebSocketEventHandlers:
    """Event handlers for WebSocket updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.event_bus = Container().get_event_bus()
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers"""
        # Subscribe to workflow events
        self.event_bus.subscribe("workflow.created", self.handle_workflow_created)
        self.event_bus.subscribe("workflow.updated", self.handle_workflow_updated)
        self.event_bus.subscribe("workflow.deleted", self.handle_workflow_deleted)
        self.event_bus.subscribe("workflow.status_changed", self.handle_workflow_status_changed)
        
        # Subscribe to node events
        self.event_bus.subscribe("node.added", self.handle_node_added)
        self.event_bus.subscribe("node.updated", self.handle_node_updated)
        self.event_bus.subscribe("node.deleted", self.handle_node_deleted)
        self.event_bus.subscribe("node.status_changed", self.handle_node_status_changed)
    
    async def handle_workflow_created(self, event_data: Dict[str, Any]):
        """Handle workflow created event"""
        workflow_id = event_data.get("workflow_id")
        if workflow_id:
            message = WebSocketMessage.create_workflow_update(
                workflow_id,
                "created",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_workflow_updated(self, event_data: Dict[str, Any]):
        """Handle workflow updated event"""
        workflow_id = event_data.get("workflow_id")
        if workflow_id:
            message = WebSocketMessage.create_workflow_update(
                workflow_id,
                "updated",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_workflow_deleted(self, event_data: Dict[str, Any]):
        """Handle workflow deleted event"""
        workflow_id = event_data.get("workflow_id")
        if workflow_id:
            message = WebSocketMessage.create_workflow_update(
                workflow_id,
                "deleted",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_workflow_status_changed(self, event_data: Dict[str, Any]):
        """Handle workflow status changed event"""
        workflow_id = event_data.get("workflow_id")
        if workflow_id:
            message = WebSocketMessage.create_workflow_update(
                workflow_id,
                "status_changed",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_node_added(self, event_data: Dict[str, Any]):
        """Handle node added event"""
        workflow_id = event_data.get("workflow_id")
        node_id = event_data.get("node_id")
        if workflow_id and node_id:
            message = WebSocketMessage.create_node_update(
                workflow_id,
                node_id,
                "added",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_node_updated(self, event_data: Dict[str, Any]):
        """Handle node updated event"""
        workflow_id = event_data.get("workflow_id")
        node_id = event_data.get("node_id")
        if workflow_id and node_id:
            message = WebSocketMessage.create_node_update(
                workflow_id,
                node_id,
                "updated",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_node_deleted(self, event_data: Dict[str, Any]):
        """Handle node deleted event"""
        workflow_id = event_data.get("workflow_id")
        node_id = event_data.get("node_id")
        if workflow_id and node_id:
            message = WebSocketMessage.create_node_update(
                workflow_id,
                node_id,
                "deleted",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)
    
    async def handle_node_status_changed(self, event_data: Dict[str, Any]):
        """Handle node status changed event"""
        workflow_id = event_data.get("workflow_id")
        node_id = event_data.get("node_id")
        if workflow_id and node_id:
            message = WebSocketMessage.create_node_update(
                workflow_id,
                node_id,
                "status_changed",
                event_data
            )
            await self.connection_manager.send_workflow_update(workflow_id, message)


# Initialize event handlers
websocket_event_handlers = WebSocketEventHandlers(manager)


# Utility functions
async def broadcast_system_notification(notification_type: str, message: str, data: Optional[Dict[str, Any]] = None):
    """Broadcast system notification to all connections"""
    notification = WebSocketMessage.create_system_notification(notification_type, message, data)
    await manager.broadcast(notification)


async def send_workflow_update(workflow_id: str, update_type: str, data: Dict[str, Any]):
    """Send workflow update to subscribed connections"""
    message = WebSocketMessage.create_workflow_update(workflow_id, update_type, data)
    await manager.send_workflow_update(workflow_id, message)


async def send_node_update(workflow_id: str, node_id: str, update_type: str, data: Dict[str, Any]):
    """Send node update to subscribed connections"""
    message = WebSocketMessage.create_node_update(workflow_id, node_id, update_type, data)
    await manager.send_workflow_update(workflow_id, message)


async def send_user_notification(user_id: str, notification_type: str, message: str, data: Optional[Dict[str, Any]] = None):
    """Send notification to specific user"""
    notification = WebSocketMessage.create_system_notification(notification_type, message, data)
    await manager.send_user_message(user_id, notification)


def get_connection_stats() -> Dict[str, Any]:
    """Get WebSocket connection statistics"""
    return manager.get_connection_stats()
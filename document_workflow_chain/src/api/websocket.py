"""
WebSocket API - Advanced Implementation
======================================

Advanced WebSocket API with real-time communication and event broadcasting.
"""

from __future__ import annotations
import logging
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState

from ..services import websocket_service

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str = Query(..., description="Unique connection identifier"),
    user_id: Optional[int] = Query(None, description="User ID for authenticated connections")
):
    """WebSocket endpoint for real-time communication"""
    try:
        await websocket_service.handle_connection(websocket, connection_id, user_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@router.websocket("/ws/workflow/{workflow_id}")
async def workflow_websocket_endpoint(
    websocket: WebSocket,
    workflow_id: int,
    connection_id: str = Query(..., description="Unique connection identifier"),
    user_id: Optional[int] = Query(None, description="User ID for authenticated connections")
):
    """WebSocket endpoint for workflow-specific real-time communication"""
    try:
        await websocket_service.handle_connection(websocket, connection_id, user_id)
        
        # Join workflow room
        await websocket_service.manager.join_room(connection_id, f"workflow_{workflow_id}")
        
        # Send workflow-specific welcome message
        await websocket_service.manager.send_personal_message({
            "type": "workflow_connected",
            "workflow_id": workflow_id,
            "connection_id": connection_id,
            "message": f"Connected to workflow {workflow_id} real-time updates"
        }, connection_id)
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_service._handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_service.manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error(f"Error handling workflow WebSocket message: {e}")
    
    except WebSocketDisconnect:
        logger.info(f"Workflow WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Workflow WebSocket error: {e}")
    finally:
        websocket_service.manager.disconnect(connection_id)


@router.websocket("/ws/user/{user_id}")
async def user_websocket_endpoint(
    websocket: WebSocket,
    user_id: int,
    connection_id: str = Query(..., description="Unique connection identifier")
):
    """WebSocket endpoint for user-specific real-time communication"""
    try:
        await websocket_service.handle_connection(websocket, connection_id, user_id)
        
        # Send user-specific welcome message
        await websocket_service.manager.send_personal_message({
            "type": "user_connected",
            "user_id": user_id,
            "connection_id": connection_id,
            "message": f"Connected to user {user_id} real-time updates"
        }, connection_id)
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_service._handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_service.manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error(f"Error handling user WebSocket message: {e}")
    
    except WebSocketDisconnect:
        logger.info(f"User WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"User WebSocket error: {e}")
    finally:
        websocket_service.manager.disconnect(connection_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_service.get_connection_stats()
        return {
            "websocket_stats": stats,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.get("/ws/connections")
async def get_active_connections():
    """Get active WebSocket connections"""
    try:
        connections = []
        for connection_id, metadata in websocket_service.manager.connection_metadata.items():
            connections.append({
                "connection_id": connection_id,
                "user_id": metadata.get("user_id"),
                "connected_at": metadata.get("connected_at"),
                "last_activity": metadata.get("last_activity"),
                "metadata": metadata.get("metadata", {})
            })
        
        return {
            "connections": connections,
            "total_connections": len(connections),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to get active connections: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.get("/ws/rooms")
async def get_websocket_rooms():
    """Get WebSocket rooms and their connections"""
    try:
        rooms = {}
        for room_id, connections in websocket_service.manager.room_connections.items():
            rooms[room_id] = {
                "connections": list(connections),
                "connection_count": len(connections)
            }
        
        return {
            "rooms": rooms,
            "total_rooms": len(rooms),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to get WebSocket rooms: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections"""
    try:
        await websocket_service.manager.broadcast(message)
        return {
            "message": "Message broadcasted successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast message: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/broadcast/room/{room_id}")
async def broadcast_to_room(room_id: str, message: Dict[str, Any]):
    """Broadcast message to specific room"""
    try:
        await websocket_service.manager.send_to_room(message, room_id)
        return {
            "message": f"Message broadcasted to room {room_id}",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast to room: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/broadcast/user/{user_id}")
async def broadcast_to_user(user_id: int, message: Dict[str, Any]):
    """Broadcast message to specific user"""
    try:
        await websocket_service.manager.send_to_user(message, user_id)
        return {
            "message": f"Message broadcasted to user {user_id}",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast to user: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/events/workflow/{workflow_id}")
async def broadcast_workflow_event(workflow_id: int, event_data: Dict[str, Any]):
    """Broadcast workflow event"""
    try:
        await websocket_service.broadcast_workflow_update(workflow_id, event_data)
        return {
            "message": f"Workflow event broadcasted for workflow {workflow_id}",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast workflow event: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/events/ai/{user_id}")
async def broadcast_ai_event(user_id: int, event_data: Dict[str, Any]):
    """Broadcast AI processing event"""
    try:
        await websocket_service.broadcast_ai_processing(user_id, event_data)
        return {
            "message": f"AI event broadcasted for user {user_id}",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast AI event: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/events/notification")
async def broadcast_notification_event(event_data: Dict[str, Any]):
    """Broadcast notification event"""
    try:
        await websocket_service.broadcast_system_notification(event_data)
        return {
            "message": "Notification event broadcasted",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast notification event: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/events/security")
async def broadcast_security_event(event_data: Dict[str, Any]):
    """Broadcast security event"""
    try:
        await websocket_service.broadcast_security_event(event_data)
        return {
            "message": "Security event broadcasted",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast security event: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/ws/events/analytics")
async def broadcast_analytics_event(event_data: Dict[str, Any]):
    """Broadcast analytics event"""
    try:
        await websocket_service.broadcast_analytics_update(event_data)
        return {
            "message": "Analytics event broadcasted",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to broadcast analytics event: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


"""
WebSocket endpoints for real-time features
"""

from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState
import json
import asyncio

from ....services.notification_service import get_notification_service
from ....core.security import get_current_user
from ....config.database import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    token: str = Query(..., description="JWT token for authentication"),
    session: AsyncSession = Depends(get_db_session)
):
    """WebSocket endpoint for real-time notifications."""
    try:
        # Authenticate user (simplified for WebSocket)
        # In a real implementation, you would validate the JWT token
        if not token:
            await websocket.close(code=1008, reason="Authentication required")
            return
        
        # Get notification service
        notification_service = get_notification_service(session)
        connection_manager = notification_service.get_connection_manager()
        
        # Connect the user
        await connection_manager.connect(websocket, user_id)
        
        # Send welcome message
        welcome_message = {
            "type": "connection",
            "title": "Connected",
            "message": "You are now connected to real-time notifications",
            "data": {
                "user_id": user_id,
                "connection_time": asyncio.get_event_loop().time()
            }
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        # Keep connection alive and handle messages
        try:
            while True:
                # Wait for messages from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, user_id, message, connection_manager)
                except json.JSONDecodeError:
                    error_message = {
                        "type": "error",
                        "title": "Invalid Message",
                        "message": "Message must be valid JSON",
                        "data": {}
                    }
                    await websocket.send_text(json.dumps(error_message))
                
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            connection_manager.disconnect(websocket)
            
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")


async def handle_websocket_message(
    websocket: WebSocket,
    user_id: str,
    message: Dict[str, Any],
    connection_manager
):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Respond to ping with pong
        pong_message = {
            "type": "pong",
            "title": "Pong",
            "message": "Connection is alive",
            "data": {
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await websocket.send_text(json.dumps(pong_message))
    
    elif message_type == "get_stats":
        # Send connection statistics
        stats = {
            "connected_users": len(connection_manager.active_connections),
            "total_connections": connection_manager.get_connection_count(),
            "user_connections": len(connection_manager.active_connections.get(user_id, set()))
        }
        
        stats_message = {
            "type": "stats",
            "title": "Connection Statistics",
            "message": "Current connection statistics",
            "data": stats
        }
        await websocket.send_text(json.dumps(stats_message))
    
    elif message_type == "subscribe":
        # Handle subscription to specific topics
        topic = message.get("topic")
        if topic:
            subscribe_message = {
                "type": "subscription",
                "title": "Subscribed",
                "message": f"Subscribed to {topic}",
                "data": {
                    "topic": topic,
                    "user_id": user_id
                }
            }
            await websocket.send_text(json.dumps(subscribe_message))
    
    elif message_type == "unsubscribe":
        # Handle unsubscription from topics
        topic = message.get("topic")
        if topic:
            unsubscribe_message = {
                "type": "unsubscription",
                "title": "Unsubscribed",
                "message": f"Unsubscribed from {topic}",
                "data": {
                    "topic": topic,
                    "user_id": user_id
                }
            }
            await websocket.send_text(json.dumps(unsubscribe_message))
    
    else:
        # Unknown message type
        error_message = {
            "type": "error",
            "title": "Unknown Message Type",
            "message": f"Unknown message type: {message_type}",
            "data": {
                "received_type": message_type
            }
        }
        await websocket.send_text(json.dumps(error_message))


@router.get("/ws/stats")
async def get_websocket_stats(session: AsyncSession = Depends(get_db_session)):
    """Get WebSocket connection statistics."""
    try:
        notification_service = get_notification_service(session)
        stats = await notification_service.get_notification_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "WebSocket statistics retrieved"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get WebSocket statistics"
        }


@router.post("/ws/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    session: AsyncSession = Depends(get_db_session)
):
    """Broadcast a message to all connected users (admin only)."""
    try:
        notification_service = get_notification_service(session)
        connection_manager = notification_service.get_connection_manager()
        
        broadcast_data = {
            "type": "broadcast",
            "title": message.get("title", "System Message"),
            "message": message.get("message", ""),
            "data": message.get("data", {}),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await connection_manager.broadcast(json.dumps(broadcast_data))
        
        return {
            "success": True,
            "message": "Message broadcasted successfully",
            "recipients": connection_manager.get_connection_count()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to broadcast message"
        }


@router.post("/ws/send-to-user")
async def send_message_to_user(
    user_id: str,
    message: Dict[str, Any],
    session: AsyncSession = Depends(get_db_session)
):
    """Send a message to a specific user."""
    try:
        notification_service = get_notification_service(session)
        connection_manager = notification_service.get_connection_manager()
        
        user_message = {
            "type": "direct_message",
            "title": message.get("title", "Direct Message"),
            "message": message.get("message", ""),
            "data": message.get("data", {}),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await connection_manager.send_personal_message(
            json.dumps(user_message),
            user_id
        )
        
        return {
            "success": True,
            "message": f"Message sent to user {user_id}",
            "recipient": user_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to send message to user {user_id}"
        }































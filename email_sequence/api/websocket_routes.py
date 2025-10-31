"""
WebSocket Routes for Real-Time Email Sequence System

This module provides WebSocket endpoints for real-time communication,
live analytics, and event streaming.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, Path
from fastapi.responses import HTMLResponse

from .schemas import ErrorResponse
from ..core.real_time_engine import (
    real_time_engine,
    EventType,
    ConnectionType,
    RealTimeEvent
)
from ..core.dependencies import get_current_user
from ..core.exceptions import RealTimeProcessingError

logger = logging.getLogger(__name__)

# WebSocket router
websocket_router = APIRouter(
    prefix="/api/v1/ws",
    tags=["WebSocket"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


@websocket_router.websocket("/connect/{connection_type}")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_type: str = Path(..., description="Type of connection"),
    user_id: str = Query(..., description="User ID"),
    token: str = Query(..., description="Authentication token")
):
    """
    WebSocket endpoint for real-time connections.
    
    Args:
        websocket: WebSocket connection
        connection_type: Type of connection (dashboard, analytics, monitoring, admin)
        user_id: User ID
        token: Authentication token
    """
    connection_id = None
    
    try:
        # Validate connection type
        try:
            conn_type = ConnectionType(connection_type)
        except ValueError:
            await websocket.close(code=4000, reason="Invalid connection type")
            return
        
        # TODO: Validate authentication token
        # For now, we'll accept any token
        
        # Connect to real-time engine
        connection_id = await real_time_engine.connect_websocket(
            websocket=websocket,
            connection_type=conn_type,
            user_id=user_id
        )
        
        logger.info(f"WebSocket connected: {connection_id} ({connection_type})")
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Process message
                await _handle_websocket_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat()
                }))
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        if connection_id:
            await real_time_engine.disconnect_websocket(connection_id)
    finally:
        if connection_id:
            await real_time_engine.disconnect_websocket(connection_id)


@websocket_router.get("/dashboard", response_class=HTMLResponse)
async def websocket_dashboard():
    """
    WebSocket dashboard for testing connections.
    
    Returns:
        HTML page for WebSocket testing
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Sequence Real-Time Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background-color: #d4edda; color: #155724; }
            .disconnected { background-color: #f8d7da; color: #721c24; }
            .messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
            .message { margin: 5px 0; padding: 5px; background-color: #f8f9fa; border-radius: 3px; }
            .controls { margin: 10px 0; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            input, select { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Email Sequence Real-Time Dashboard</h1>
            
            <div id="status" class="status disconnected">
                Disconnected
            </div>
            
            <div class="controls">
                <input type="text" id="userId" placeholder="User ID" value="test_user">
                <select id="connectionType">
                    <option value="dashboard">Dashboard</option>
                    <option value="analytics">Analytics</option>
                    <option value="monitoring">Monitoring</option>
                    <option value="admin">Admin</option>
                </select>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="clearMessages()">Clear Messages</button>
            </div>
            
            <div class="controls">
                <h3>Subscribe to Events:</h3>
                <label><input type="checkbox" value="email_sent"> Email Sent</label>
                <label><input type="checkbox" value="email_opened"> Email Opened</label>
                <label><input type="checkbox" value="email_clicked"> Email Clicked</label>
                <label><input type="checkbox" value="email_bounced"> Email Bounced</label>
                <label><input type="checkbox" value="analytics_update"> Analytics Update</label>
                <label><input type="checkbox" value="performance_alert"> Performance Alert</label>
                <button onclick="subscribe()">Subscribe</button>
            </div>
            
            <div class="controls">
                <h3>Send Test Event:</h3>
                <select id="testEventType">
                    <option value="email_sent">Email Sent</option>
                    <option value="email_opened">Email Opened</option>
                    <option value="email_clicked">Email Clicked</option>
                    <option value="analytics_update">Analytics Update</option>
                </select>
                <input type="text" id="testSequenceId" placeholder="Sequence ID" value="123e4567-e89b-12d3-a456-426614174000">
                <button onclick="sendTestEvent()">Send Test Event</button>
            </div>
            
            <h3>Messages:</h3>
            <div id="messages" class="messages"></div>
        </div>
        
        <script>
            let ws = null;
            let connectionId = null;
            
            function connect() {
                const userId = document.getElementById('userId').value;
                const connectionType = document.getElementById('connectionType').value;
                
                if (!userId) {
                    alert('Please enter a User ID');
                    return;
                }
                
                const wsUrl = `ws://localhost:8000/api/v1/ws/connect/${connectionType}?user_id=${userId}&token=test_token`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    updateStatus('Connected', 'connected');
                    addMessage('Connected to WebSocket', 'info');
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    addMessage(JSON.stringify(message, null, 2), 'message');
                };
                
                ws.onclose = function(event) {
                    updateStatus('Disconnected', 'disconnected');
                    addMessage('WebSocket connection closed', 'info');
                };
                
                ws.onerror = function(error) {
                    updateStatus('Error', 'disconnected');
                    addMessage('WebSocket error: ' + error, 'error');
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function subscribe() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('Not connected to WebSocket');
                    return;
                }
                
                const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
                const eventTypes = Array.from(checkboxes).map(cb => cb.value);
                
                if (eventTypes.length === 0) {
                    alert('Please select at least one event type');
                    return;
                }
                
                const message = {
                    type: 'subscribe',
                    event_types: eventTypes,
                    filters: {}
                };
                
                ws.send(JSON.stringify(message));
                addMessage('Subscribed to: ' + eventTypes.join(', '), 'info');
            }
            
            function sendTestEvent() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('Not connected to WebSocket');
                    return;
                }
                
                const eventType = document.getElementById('testEventType').value;
                const sequenceId = document.getElementById('testSequenceId').value;
                
                const message = {
                    type: 'test_event',
                    event_type: eventType,
                    sequence_id: sequenceId,
                    data: {
                        test: true,
                        timestamp: new Date().toISOString()
                    }
                };
                
                ws.send(JSON.stringify(message));
                addMessage('Sent test event: ' + eventType, 'info');
            }
            
            function updateStatus(text, className) {
                const status = document.getElementById('status');
                status.textContent = text;
                status.className = 'status ' + className;
            }
            
            function addMessage(text, type) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                messageDiv.innerHTML = `<strong>[${new Date().toLocaleTimeString()}]</strong> ${text}`;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function clearMessages() {
                document.getElementById('messages').innerHTML = '';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@websocket_router.get("/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.
    
    Returns:
        Dictionary with WebSocket statistics
    """
    try:
        stats = await real_time_engine.get_connection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise RealTimeProcessingError(f"Failed to get WebSocket stats: {e}")


@websocket_router.post("/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    connection_type: Optional[str] = Query(default=None, description="Connection type filter"),
    user_id: Optional[str] = Query(default=None, description="User ID filter")
):
    """
    Broadcast message to WebSocket connections.
    
    Args:
        message: Message to broadcast
        connection_type: Optional connection type filter
        user_id: Optional user ID filter
        
    Returns:
        Broadcast result
    """
    try:
        conn_type = None
        if connection_type:
            try:
                conn_type = ConnectionType(connection_type)
            except ValueError:
                raise RealTimeProcessingError(f"Invalid connection type: {connection_type}")
        
        await real_time_engine.broadcast_to_connections(
            message=message,
            connection_type=conn_type,
            user_id=user_id
        )
        
        return {
            "status": "success",
            "message": "Message broadcasted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise RealTimeProcessingError(f"Failed to broadcast message: {e}")


@websocket_router.post("/event")
async def publish_event(
    event_type: str = Query(..., description="Event type"),
    sequence_id: str = Query(..., description="Sequence ID"),
    subscriber_id: Optional[str] = Query(default=None, description="Subscriber ID"),
    campaign_id: Optional[str] = Query(default=None, description="Campaign ID"),
    data: Dict[str, Any] = {}
):
    """
    Publish a real-time event.
    
    Args:
        event_type: Type of event
        sequence_id: Sequence ID
        subscriber_id: Optional subscriber ID
        campaign_id: Optional campaign ID
        data: Event data
        
    Returns:
        Event publication result
    """
    try:
        # Validate event type
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            raise RealTimeProcessingError(f"Invalid event type: {event_type}")
        
        # Create event
        event = RealTimeEvent(
            event_type=event_type_enum,
            sequence_id=UUID(sequence_id),
            subscriber_id=UUID(subscriber_id) if subscriber_id else None,
            campaign_id=UUID(campaign_id) if campaign_id else None,
            data=data
        )
        
        # Publish event
        await real_time_engine.publish_event(event)
        
        return {
            "status": "success",
            "message": "Event published successfully",
            "event_id": str(event.sequence_id),
            "timestamp": event.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error publishing event: {e}")
        raise RealTimeProcessingError(f"Failed to publish event: {e}")


# Helper function to handle WebSocket messages
async def _handle_websocket_message(connection_id: str, message: Dict[str, Any]) -> None:
    """
    Handle incoming WebSocket message.
    
    Args:
        connection_id: Connection ID
        message: Message data
    """
    try:
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Handle subscription request
            event_types = message.get("event_types", [])
            filters = message.get("filters", {})
            
            # Convert string event types to enum
            event_type_enums = []
            for event_type_str in event_types:
                try:
                    event_type_enums.append(EventType(event_type_str))
                except ValueError:
                    logger.warning(f"Invalid event type: {event_type_str}")
            
            if event_type_enums:
                await real_time_engine.subscribe_to_events(
                    connection_id=connection_id,
                    event_types=event_type_enums,
                    filters=filters
                )
        
        elif message_type == "ping":
            # Handle ping message
            await real_time_engine._send_to_connection(connection_id, {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif message_type == "test_event":
            # Handle test event
            event_type = message.get("event_type")
            sequence_id = message.get("sequence_id")
            data = message.get("data", {})
            
            if event_type and sequence_id:
                try:
                    event = RealTimeEvent(
                        event_type=EventType(event_type),
                        sequence_id=UUID(sequence_id),
                        data=data
                    )
                    await real_time_engine.publish_event(event)
                except ValueError:
                    logger.warning(f"Invalid event type for test: {event_type}")
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")


# Error handlers for WebSocket routes
@websocket_router.exception_handler(RealTimeProcessingError)
async def real_time_error_handler(request, exc):
    """Handle real-time processing errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=f"Real-time processing error: {exc.message}",
            error_code="REAL_TIME_ERROR"
        ).dict()
    )































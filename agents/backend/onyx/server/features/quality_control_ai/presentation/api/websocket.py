"""
WebSocket Handlers

WebSocket endpoints for real-time communication.
"""

from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
from typing import List

from ...application.services import InspectionApplicationService
from ...application.dto import InspectionRequest

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast: {str(e)}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


async def websocket_inspection_stream(
    websocket: WebSocket,
    service: InspectionApplicationService
):
    """
    WebSocket endpoint for real-time inspection streaming.
    
    Args:
        websocket: WebSocket connection
        service: Inspection application service
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "inspect":
                # Inspect image
                try:
                    request = InspectionRequest(
                        image_data=data.get("image_data"),
                        image_format=data.get("image_format", "base64"),
                        include_visualization=data.get("include_visualization", False),
                    )
                    
                    response = service.inspect_image(request)
                    
                    # Send response
                    await manager.send_personal_message({
                        "type": "inspection_result",
                        "data": response.to_dict(),
                    }, websocket)
                
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e),
                    }, websocket)
            
            elif message_type == "ping":
                # Respond to ping
                await manager.send_personal_message({
                    "type": "pong",
                }, websocket)
            
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        manager.disconnect(websocket)




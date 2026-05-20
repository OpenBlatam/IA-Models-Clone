"""
WebSocket Routes for Real-time Updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from uuid import UUID
import logging

from ..services.realtime import get_websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: UUID):
    """
    WebSocket endpoint for real-time video generation updates
    
    Args:
        websocket: WebSocket connection
        video_id: Video ID to track
    """
    await websocket.accept()
    
    ws_manager = get_websocket_manager()
    ws_manager.connect(video_id, websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or handle client messages
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        ws_manager.disconnect(video_id, websocket)
        logger.info(f"WebSocket disconnected for video {video_id}")


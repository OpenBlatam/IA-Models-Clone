"""
WebSocket Manager for Real-time Updates
"""

from typing import Dict, Set, Optional
from uuid import UUID
import json
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.connections: Dict[UUID, Set] = {}  # video_id -> set of connections
        self.user_connections: Dict[str, Set] = {}  # user_id -> set of connections
    
    def connect(self, video_id: UUID, websocket) -> bool:
        """
        Register WebSocket connection for video updates
        
        Args:
            video_id: Video ID to track
            websocket: WebSocket connection
            
        Returns:
            True if connected
        """
        if video_id not in self.connections:
            self.connections[video_id] = set()
        
        self.connections[video_id].add(websocket)
        logger.debug(f"WebSocket connected for video {video_id}")
        return True
    
    def disconnect(self, video_id: UUID, websocket) -> bool:
        """Disconnect WebSocket"""
        if video_id in self.connections:
            self.connections[video_id].discard(websocket)
            if not self.connections[video_id]:
                del self.connections[video_id]
        logger.debug(f"WebSocket disconnected for video {video_id}")
        return True
    
    async def send_update(self, video_id: UUID, message: dict):
        """
        Send update to all connections for a video
        
        Args:
            video_id: Video ID
            message: Message to send
        """
        if video_id not in self.connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for websocket in self.connections[video_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {str(e)}")
                disconnected.add(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            self.connections[video_id].discard(ws)
    
    async def broadcast_progress(self, video_id: UUID, progress: float, status: str, message: str):
        """Broadcast progress update"""
        await self.send_update(video_id, {
            "type": "progress",
            "video_id": str(video_id),
            "progress": progress,
            "status": status,
            "message": message,
        })
    
    async def broadcast_completion(self, video_id: UUID, video_url: str):
        """Broadcast completion"""
        await self.send_update(video_id, {
            "type": "completed",
            "video_id": str(video_id),
            "video_url": video_url,
        })
    
    async def broadcast_error(self, video_id: UUID, error: str):
        """Broadcast error"""
        await self.send_update(video_id, {
            "type": "error",
            "video_id": str(video_id),
            "error": error,
        })


_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance (singleton)"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


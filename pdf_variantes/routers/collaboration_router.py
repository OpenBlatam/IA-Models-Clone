"""Collaboration router with functional approach."""

from fastapi import APIRouter, Depends, Query, Path, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

from ..dependencies import get_pdf_service, get_current_user
from ..exceptions import PDFNotFoundError
from ..schemas import CollaborationSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collaboration", tags=["Collaboration"])


class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: Dict[str, list] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
    
    async def broadcast(self, message: str, session_id: str):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message: {e}")


manager = ConnectionManager()


@router.post("/sessions")
async def create_session(
    collaboration_data: CollaborationSchema,
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create collaboration session."""
    session = pdf_service.advanced.create_collaboration_session(
        collaboration_data.session_name,
        current_user.get("user_id", "anonymous")
    )
    
    return {
        "session_id": session.session_id,
        "session_name": collaboration_data.session_name,
        "created_by": current_user.get("user_id"),
        "created_at": datetime.utcnow().isoformat()
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str = Path(...),
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get collaboration session."""
    session = pdf_service.advanced.get_collaboration_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "session_name": session.session_name,
        "created_by": session.created_by,
        "participants": session.participants
    }


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Path(...),
    user_id: str = Query(...)
):
    """WebSocket for real-time collaboration."""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await manager.broadcast(
                json.dumps({
                    "type": "collaboration_message",
                    "user_id": user_id,
                    "data": message,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                session_id
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)

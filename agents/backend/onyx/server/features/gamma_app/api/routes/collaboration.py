"""
Collaboration Routes
API endpoints for collaboration sessions and real-time collaboration
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status

from ..models import CollaborationSession, User
from ..dependencies import get_collaboration_service, get_current_user
from ..error_handlers import handle_route_errors
from ...services.collaboration_service import CollaborationService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/session", response_model=CollaborationSession)
@handle_route_errors
async def create_collaboration_session(
    project_id: str = Query(..., description="Project ID"),
    session_name: str = Query(..., description="Session name"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service),
    current_user: User = Depends(get_current_user)
) -> CollaborationSession:
    """Create new collaboration session"""
    session = await collaboration_service.create_session(
        project_id=project_id,
        session_name=session_name,
        creator_id=current_user.id
    )
    return session

@router.get("/session/{session_id}")
@handle_route_errors
async def get_collaboration_session(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service),
    current_user: User = Depends(get_current_user)
) -> CollaborationSession:
    """Get collaboration session details"""
    session = await collaboration_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    return session

@router.post("/session/{session_id}/join")
@handle_route_errors
async def join_collaboration_session(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Join collaboration session"""
    success = await collaboration_service.join_session(
        session_id=session_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not join session"
        )
    return {"message": "Successfully joined session"}

@router.post("/session/{session_id}/leave")
@handle_route_errors
async def leave_collaboration_session(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Leave collaboration session"""
    success = await collaboration_service.leave_session(
        session_id=session_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not leave session"
        )
    return {"message": "Successfully left session"}

@router.get("/session/{session_id}/participants")
@handle_route_errors
async def get_session_participants(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get session participants"""
    participants = await collaboration_service.get_session_participants(session_id)
    return {"participants": participants}

@router.websocket("/session/{session_id}/ws")
async def collaboration_websocket(
    websocket,
    session_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> None:
    """WebSocket endpoint for real-time collaboration"""
    await collaboration_service.handle_websocket_connection(websocket, session_id)








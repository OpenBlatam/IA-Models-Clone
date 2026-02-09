"""
API de Colaboración

Endpoints para:
- Crear sesiones de colaboración
- Unirse/salir de sesiones
- Enviar eventos
- Obtener historial
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.collaboration import get_collaboration_service
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/collaboration",
    tags=["collaboration"]
)


@router.post("/sessions")
async def create_session(
    project_id: str = Body(..., description="ID del proyecto"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea una sesión de colaboración.
    """
    try:
        import uuid
        session_id = str(uuid.uuid4())
        user_id = current_user.get("user_id", "unknown")
        
        service = get_collaboration_service()
        session = service.create_session(session_id, project_id, user_id)
        
        return {
            "session_id": session.session_id,
            "project_id": session.project_id,
            "owner_id": session.owner_id,
            "participants": list(session.participants),
            "created_at": session.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )


@router.post("/sessions/{session_id}/join")
async def join_session(
    session_id: str,
    permission: str = Body("editor", description="Permiso (viewer, editor)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Une un usuario a una sesión.
    """
    try:
        user_id = current_user.get("user_id", "unknown")
        service = get_collaboration_service()
        
        success = service.join_session(session_id, user_id, permission)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or inactive"
            )
        
        return {"message": "Joined session", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error joining session: {str(e)}"
        )


@router.post("/sessions/{session_id}/events")
async def add_event(
    session_id: str,
    event_type: str = Body(..., description="Tipo de evento"),
    data: Dict[str, Any] = Body(..., description="Datos del evento"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Agrega un evento de colaboración.
    """
    try:
        user_id = current_user.get("user_id", "unknown")
        service = get_collaboration_service()
        
        service.add_event(session_id, user_id, event_type, data)
        
        return {"message": "Event added", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error adding event: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding event: {str(e)}"
        )


@router.get("/sessions/{session_id}/events")
async def get_events(
    session_id: str,
    since: Optional[str] = Query(None, description="Fecha desde (ISO format)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene eventos de una sesión.
    """
    try:
        from datetime import datetime
        service = get_collaboration_service()
        
        since_date = None
        if since:
            since_date = datetime.fromisoformat(since)
        
        events = service.get_events(session_id, since_date)
        
        return {
            "events": [
                {
                    "event_type": e.event_type,
                    "user_id": e.user_id,
                    "data": e.data,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in events
            ]
        }
    except Exception as e:
        logger.error(f"Error getting events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving events: {str(e)}"
        )


@router.get("/sessions")
async def get_user_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene sesiones de un usuario.
    """
    try:
        user_id = current_user.get("user_id", "unknown")
        service = get_collaboration_service()
        
        sessions = service.get_user_sessions(user_id)
        
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "project_id": s.project_id,
                    "owner_id": s.owner_id,
                    "participants": list(s.participants),
                    "active": s.active
                }
                for s in sessions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sessions: {str(e)}"
        )


"""
Audit Routes - Rutas para consultar eventos de auditoría.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from datetime import datetime

from api.utils import handle_api_errors
from api.dependencies import get_storage
from core.storage import TaskStorage
from core.services import AuditService, AuditEventType
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


@router.get("/events")
@handle_api_errors
async def get_audit_events(
    event_type: Optional[str] = Query(None, description="Filtrar por tipo de evento"),
    user: Optional[str] = Query(None, description="Filtrar por usuario"),
    limit: int = Query(100, ge=1, le=1000, description="Número máximo de eventos")
):
    """
    Obtener eventos de auditoría.
    
    Args:
        event_type: Tipo de evento a filtrar
        user: Usuario a filtrar
        limit: Número máximo de eventos
        
    Returns:
        Lista de eventos de auditoría
    """
    try:
        audit_service: AuditService = get_service("audit_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Audit service no disponible")
    
    # Validar event_type
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = AuditEventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de evento inválido: {event_type}"
            )
    
    events = audit_service.get_events(
        event_type=event_type_enum,
        user=user,
        limit=limit
    )
    
    return {
        "total": len(events),
        "events": events
    }


@router.get("/stats")
@handle_api_errors
async def get_audit_stats():
    """
    Obtener estadísticas de auditoría.
    
    Returns:
        Estadísticas de auditoría
    """
    try:
        audit_service: AuditService = get_service("audit_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Audit service no disponible")
    
    return audit_service.get_stats()




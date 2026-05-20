"""
Rutas para Auditoría
====================

Endpoints para logs de auditoría.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..core.audit_logger import get_audit_logger, AuditLogger, ActionType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/audit",
    tags=["Audit Logs"]
)


@router.get("/logs")
async def get_audit_logs(
    action_type: Optional[str] = None,
    user_id: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 100,
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Obtener logs de auditoría"""
    try:
        action_type_enum = ActionType(action_type) if action_type else None
        
        logs = audit_logger.get_logs(
            action_type_enum,
            user_id,
            document_id,
            limit
        )
        
        return {
            "total": len(logs),
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Error obteniendo logs de auditoría: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_audit_statistics(
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Obtener estadísticas de auditoría"""
    return audit_logger.get_statistics()

















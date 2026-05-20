"""
Audit Log endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.audit_log import AuditLogService, AuditAction, AuditSeverity

router = APIRouter()
audit_service = AuditLogService()


@router.post("/log")
async def log_action(
    user_id: Optional[str],
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    severity: str = "medium",
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Dict[str, Any]:
    """Registrar acción en log de auditoría"""
    try:
        action_enum = AuditAction(action)
        severity_enum = AuditSeverity(severity)
        
        log = audit_service.log_action(
            user_id, action_enum, resource_type, resource_id,
            severity_enum, details, ip_address, user_agent
        )
        
        return {
            "id": log.id,
            "action": log.action.value,
            "resource_type": log.resource_type,
            "timestamp": log.timestamp.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_audit_logs(
    user_id: str,
    limit: int = 100,
    action: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener logs de auditoría del usuario"""
    try:
        action_enum = AuditAction(action) if action else None
        logs = audit_service.get_user_audit_logs(user_id, limit, action_enum)
        return {
            "user_id": user_id,
            "logs": logs,
            "total": len(logs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_audit_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener estadísticas de auditoría"""
    try:
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        stats = audit_service.get_audit_statistics(start, end)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_audit_logs(
    query: str,
    limit: int = 50
) -> Dict[str, Any]:
    """Buscar en logs de auditoría"""
    try:
        results = audit_service.search_audit_logs(query, limit)
        return {
            "query": query,
            "results": results,
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





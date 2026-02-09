"""
Advanced Health Checks endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_health_checks import AdvancedHealthChecksService, CheckType

router = APIRouter()
health_service = AdvancedHealthChecksService()


@router.get("/system")
async def get_system_health(
    checks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Obtener salud del sistema"""
    try:
        check_types = [CheckType(c) for c in checks] if checks else None
        health = health_service.get_system_health(check_types)
        
        return {
            "overall_status": health.overall_status.value,
            "uptime_seconds": health.uptime_seconds,
            "checks": [
                {
                    "type": c.check_type.value,
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "response_time_ms": c.response_time_ms,
                }
                for c in health.checks
            ],
            "timestamp": health.timestamp.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_health_history(
    limit: int = 20
) -> Dict[str, Any]:
    """Obtener historial de salud"""
    try:
        history = health_service.get_health_history(limit)
        return {
            "history": history,
            "total": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





"""
Alerting System endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.alerting_system import AlertingSystemService, AlertSeverity

router = APIRouter()
alerting_service = AlertingSystemService()


@router.get("/active")
async def get_active_alerts(
    severity: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener alertas activas"""
    try:
        severity_enum = AlertSeverity(severity) if severity else None
        alerts = alerting_service.get_active_alerts(severity_enum)
        return {
            "alerts": alerts,
            "total": len(alerts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_alert_statistics(
    hours: int = 24
) -> Dict[str, Any]:
    """Obtener estadísticas de alertas"""
    try:
        stats = alerting_service.get_alert_statistics(hours)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acknowledge/{alert_id}")
async def acknowledge_alert(
    alert_id: str,
    user_id: str
) -> Dict[str, Any]:
    """Reconocer alerta"""
    try:
        success = alerting_service.acknowledge_alert(alert_id, user_id)
        return {
            "alert_id": alert_id,
            "acknowledged": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





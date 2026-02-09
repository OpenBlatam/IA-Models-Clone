"""
Job Alerts endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.job_alerts import JobAlertsService

router = APIRouter()
alerts_service = JobAlertsService()


@router.post("/create/{user_id}")
async def create_alert(
    user_id: str,
    keywords: List[str],
    location: Optional[str] = None,
    job_types: Optional[List[str]] = None,
    frequency: str = "daily"
) -> Dict[str, Any]:
    """Crear alerta de trabajo"""
    try:
        alert = alerts_service.create_alert(
            user_id, keywords, location, job_types, None, frequency
        )
        return {
            "id": alert.id,
            "keywords": alert.keywords,
            "location": alert.location,
            "frequency": alert.frequency,
            "active": alert.active,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_user_alerts(user_id: str) -> Dict[str, Any]:
    """Obtener alertas del usuario"""
    try:
        alerts = alerts_service.get_user_alerts(user_id)
        return {
            "alerts": [
                {
                    "id": a.id,
                    "keywords": a.keywords,
                    "location": a.location,
                    "frequency": a.frequency,
                    "active": a.active,
                    "matches_found": a.matches_found,
                }
                for a in alerts
            ],
            "total": len(alerts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check/{user_id}")
async def check_alerts(user_id: str) -> Dict[str, Any]:
    """Verificar alertas y encontrar matches"""
    try:
        matches = alerts_service.check_alerts(user_id)
        return {
            "matches": matches,
            "total_matches": sum(m["count"] for m in matches),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





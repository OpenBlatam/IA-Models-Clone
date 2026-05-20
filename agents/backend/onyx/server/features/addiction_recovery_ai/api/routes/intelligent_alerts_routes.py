"""
Intelligent alerts routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict

try:
    from services.intelligent_alerts_service import IntelligentAlertsService
except ImportError:
    from ...services.intelligent_alerts_service import IntelligentAlertsService

router = APIRouter()

intelligent_alerts = IntelligentAlertsService()


@router.post("/alerts/evaluate")
async def evaluate_alerts(
    user_id: str = Body(...),
    user_data: Dict = Body(...),
    recent_activity: List[Dict] = Body(...)
):
    """Evalúa condiciones y genera alertas"""
    try:
        alerts = intelligent_alerts.evaluate_alert_conditions(user_id, user_data, recent_activity)
        return JSONResponse(content={
            "user_id": user_id,
            "alerts": alerts,
            "total": len(alerts),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando alertas: {str(e)}")


@router.get("/alerts/active/{user_id}")
async def get_active_alerts(
    user_id: str,
    severity: Optional[str] = Query(None)
):
    """Obtiene alertas activas"""
    try:
        alerts = intelligent_alerts.get_active_alerts(user_id, severity)
        return JSONResponse(content={
            "user_id": user_id,
            "alerts": alerts,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")




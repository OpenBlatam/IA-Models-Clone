"""
Recovery plans routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from core.recovery_planner import RecoveryPlanner
except ImportError:
    from ...core.recovery_planner import RecoveryPlanner

router = APIRouter()

planner = RecoveryPlanner()


@router.post("/create-plan")
async def create_recovery_plan(
    user_id: str = Body(...),
    addiction_type: str = Body(...),
    severity: str = Body(...),
    goals: list = Body(...),
    preferences: Optional[dict] = Body(None)
):
    """Crea un plan de recuperación personalizado"""
    try:
        plan = planner.create_recovery_plan(user_id, addiction_type, severity, goals, preferences)
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando plan: {str(e)}")


@router.get("/plan/{user_id}")
async def get_recovery_plan(user_id: str):
    """Obtiene el plan de recuperación del usuario"""
    try:
        plan = planner.get_recovery_plan(user_id)
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo plan: {str(e)}")


@router.post("/update-plan")
async def update_recovery_plan(
    user_id: str = Body(...),
    plan_updates: dict = Body(...)
):
    """Actualiza el plan de recuperación"""
    try:
        updated_plan = planner.update_recovery_plan(user_id, plan_updates)
        return JSONResponse(content=updated_plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando plan: {str(e)}")


@router.get("/strategies/{addiction_type}")
async def get_strategies(
    addiction_type: str,
    severity: Optional[str] = Query(None)
):
    """Obtiene estrategias de recuperación para un tipo de adicción"""
    try:
        strategies = planner.get_strategies(addiction_type, severity)
        return JSONResponse(content={
            "addiction_type": addiction_type,
            "strategies": strategies,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estrategias: {str(e)}")




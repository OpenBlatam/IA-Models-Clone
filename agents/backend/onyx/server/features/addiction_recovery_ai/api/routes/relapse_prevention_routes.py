"""
Relapse prevention routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

try:
    from core.relapse_prevention import RelapsePrevention
except ImportError:
    from ...core.relapse_prevention import RelapsePrevention

router = APIRouter()

relapse_prevention = RelapsePrevention()


class RelapseRiskCheckRequest(BaseModel):
    user_id: str
    days_sober: int
    stress_level: int
    support_level: int
    triggers: List[str] = []
    previous_relapses: int = 0
    isolation: bool = False
    negative_thinking: bool = False
    romanticizing: bool = False
    skipping_support: bool = False


@router.post("/check-relapse-risk")
async def check_relapse_risk(request: RelapseRiskCheckRequest):
    """Evalúa el riesgo de recaída"""
    try:
        risk_data = request.dict()
        risk_assessment = relapse_prevention.check_relapse_risk(risk_data)
        return JSONResponse(content=risk_assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando riesgo: {str(e)}")


@router.get("/triggers/{user_id}")
async def get_user_triggers(user_id: str):
    """Obtiene triggers identificados del usuario"""
    try:
        triggers = relapse_prevention.get_user_triggers(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "triggers": triggers,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo triggers: {str(e)}")


@router.post("/coping-strategies")
async def get_coping_strategies(
    user_id: str = Body(...),
    trigger: Optional[str] = Body(None),
    situation: Optional[str] = Body(None)
):
    """Obtiene estrategias de afrontamiento"""
    try:
        strategies = relapse_prevention.get_coping_strategies(user_id, trigger, situation)
        return JSONResponse(content=strategies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estrategias: {str(e)}")


@router.post("/emergency-plan")
async def create_emergency_plan(
    user_id: str = Body(...),
    plan_details: dict = Body(...)
):
    """Crea un plan de emergencia para situaciones de alto riesgo"""
    try:
        plan = relapse_prevention.create_emergency_plan(user_id, plan_details)
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando plan de emergencia: {str(e)}")




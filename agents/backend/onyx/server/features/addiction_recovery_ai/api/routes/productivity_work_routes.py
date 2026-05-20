"""
Productivity and work analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.productivity_work_analysis_service import ProductivityWorkAnalysisService
except ImportError:
    from ...services.productivity_work_analysis_service import ProductivityWorkAnalysisService

router = APIRouter()

productivity = ProductivityWorkAnalysisService()


@router.post("/productivity/record-work")
async def record_work_session(
    user_id: str = Body(...),
    work_data: Dict = Body(...)
):
    """Registra sesión de trabajo"""
    try:
        session = productivity.record_work_session(user_id, work_data)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


@router.post("/productivity/analyze")
async def analyze_productivity(
    user_id: str = Body(...),
    work_sessions: Dict = Body(...)
):
    """Analiza productividad"""
    try:
        analysis = productivity.analyze_productivity(user_id, work_sessions)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando productividad: {str(e)}")




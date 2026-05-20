"""
Quality of life analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.quality_of_life_analysis_service import QualityOfLifeAnalysisService
except ImportError:
    from ...services.quality_of_life_analysis_service import QualityOfLifeAnalysisService

router = APIRouter()

quality_of_life = QualityOfLifeAnalysisService()


@router.post("/quality-of-life/assess")
async def assess_quality_of_life(
    user_id: str = Body(...),
    qol_data: Dict = Body(...)
):
    """Evalúa calidad de vida"""
    try:
        assessment = quality_of_life.assess_quality_of_life(user_id, qol_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando calidad de vida: {str(e)}")




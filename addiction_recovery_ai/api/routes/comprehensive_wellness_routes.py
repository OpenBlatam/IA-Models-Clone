"""
Comprehensive wellness analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.comprehensive_wellness_analysis_service import ComprehensiveWellnessAnalysisService
except ImportError:
    from ...services.comprehensive_wellness_analysis_service import ComprehensiveWellnessAnalysisService

router = APIRouter()

wellness_analysis = ComprehensiveWellnessAnalysisService()


@router.post("/wellness/assess-comprehensive")
async def assess_comprehensive_wellness(
    user_id: str = Body(...),
    wellness_data: Dict = Body(...)
):
    """Evalúa bienestar integral"""
    try:
        assessment = wellness_analysis.assess_comprehensive_wellness(user_id, wellness_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando bienestar: {str(e)}")




"""
Advanced data analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_data_analysis_service import AdvancedDataAnalysisService
except ImportError:
    from ...services.advanced_data_analysis_service import AdvancedDataAnalysisService

router = APIRouter()

advanced_data_analysis = AdvancedDataAnalysisService()


@router.post("/analysis/comprehensive")
async def perform_comprehensive_analysis(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    analysis_type: str = Body("full")
):
    """Realiza análisis comprensivo"""
    try:
        analysis = advanced_data_analysis.perform_comprehensive_analysis(
            user_id, data, analysis_type
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error realizando análisis: {str(e)}")


@router.post("/analysis/behavioral-patterns")
async def analyze_behavioral_patterns(
    user_id: str = Body(...),
    behavioral_data: List[Dict] = Body(...)
):
    """Analiza patrones de comportamiento"""
    try:
        patterns = advanced_data_analysis.analyze_behavioral_patterns(
            user_id, behavioral_data
        )
        return JSONResponse(content=patterns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")




"""
Behavioral analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_behavioral_analysis_service import AdvancedBehavioralAnalysisService
except ImportError:
    from ...services.advanced_behavioral_analysis_service import AdvancedBehavioralAnalysisService

router = APIRouter()

behavioral_analysis = AdvancedBehavioralAnalysisService()


@router.post("/behavioral/analyze-patterns")
async def analyze_behavioral_patterns(
    user_id: str = Body(...),
    behavioral_data: List[Dict] = Body(...),
    analysis_type: str = Body("comprehensive")
):
    """Analiza patrones de comportamiento"""
    try:
        analysis = behavioral_analysis.analyze_behavioral_patterns(
            user_id, behavioral_data, analysis_type
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/behavioral/detect-anomalies")
async def detect_behavioral_anomalies(
    user_id: str = Body(...),
    current_behavior: Dict = Body(...),
    historical_patterns: List[Dict] = Body(...)
):
    """Detecta anomalías de comportamiento"""
    try:
        anomalies = behavioral_analysis.detect_behavioral_anomalies(
            user_id, current_behavior, historical_patterns
        )
        return JSONResponse(content=anomalies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando anomalías: {str(e)}")




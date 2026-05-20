"""
Advanced risk analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_risk_analysis_service import AdvancedRiskAnalysisService
except ImportError:
    from ...services.advanced_risk_analysis_service import AdvancedRiskAnalysisService

router = APIRouter()

advanced_risk = AdvancedRiskAnalysisService()


@router.post("/risk/assess-comprehensive")
async def assess_comprehensive_risk(
    user_id: str = Body(...),
    risk_factors: Dict = Body(...)
):
    """Evalúa riesgo comprensivo"""
    try:
        assessment = advanced_risk.assess_comprehensive_risk(user_id, risk_factors)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando riesgo: {str(e)}")


@router.post("/risk/predict-relapse")
async def predict_relapse_risk(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice riesgo de recaída"""
    try:
        prediction = advanced_risk.predict_relapse_risk(
            user_id, current_state, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")




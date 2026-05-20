"""
Social network analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.advanced_social_network_analysis_service import AdvancedSocialNetworkAnalysisService
except ImportError:
    from ...services.advanced_social_network_analysis_service import AdvancedSocialNetworkAnalysisService

router = APIRouter()

social_network_analysis = AdvancedSocialNetworkAnalysisService()


@router.post("/social-network/analyze")
async def analyze_social_network(
    user_id: str = Body(...),
    network_data: Dict = Body(...)
):
    """Analiza red social"""
    try:
        analysis = social_network_analysis.analyze_social_network(
            user_id, network_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando red: {str(e)}")


@router.post("/social-network/predict-influence")
async def predict_social_influence(
    user_id: str = Body(...),
    network_data: Dict = Body(...)
):
    """Predice influencia social"""
    try:
        prediction = social_network_analysis.predict_social_influence(
            user_id, network_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo influencia: {str(e)}")




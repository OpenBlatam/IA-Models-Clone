"""
Genetic analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.genetic_predisposition_service import GeneticPredispositionService
except ImportError:
    from ...services.genetic_predisposition_service import GeneticPredispositionService

router = APIRouter()

genetic_analysis = GeneticPredispositionService()


@router.post("/genetic/analyze")
async def analyze_genetic_data(
    user_id: str = Body(...),
    genetic_data: Dict = Body(...)
):
    """Analiza datos genéticos"""
    try:
        analysis = genetic_analysis.analyze_genetic_data(user_id, genetic_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos genéticos: {str(e)}")


@router.post("/genetic/predict-risk")
async def predict_genetic_risk(
    user_id: str = Body(...),
    genetic_profile: Dict = Body(...),
    addiction_type: str = Body(...)
):
    """Predice riesgo genético"""
    try:
        prediction = genetic_analysis.predict_genetic_risk(
            user_id, genetic_profile, addiction_type
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")




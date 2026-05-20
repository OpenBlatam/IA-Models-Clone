"""
Advanced relationship analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_relationship_analysis_service import AdvancedRelationshipAnalysisService
except ImportError:
    from ...services.advanced_relationship_analysis_service import AdvancedRelationshipAnalysisService

router = APIRouter()

relationship_analysis = AdvancedRelationshipAnalysisService()


@router.post("/relationships/analyze")
async def analyze_relationships(
    user_id: str = Body(...),
    relationships: List[Dict] = Body(...)
):
    """Analiza relaciones"""
    try:
        analysis = relationship_analysis.analyze_relationships(user_id, relationships)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando relaciones: {str(e)}")


@router.post("/relationships/assess-impact")
async def assess_relationship_impact(
    user_id: str = Body(...),
    relationships: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Evalúa impacto de relaciones en recuperación"""
    try:
        impact = relationship_analysis.assess_relationship_impact(
            user_id, relationships, recovery_data
        )
        return JSONResponse(content=impact)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando impacto: {str(e)}")




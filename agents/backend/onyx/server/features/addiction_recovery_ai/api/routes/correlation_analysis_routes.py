"""
Correlation analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List

try:
    from services.advanced_correlation_analysis_service import AdvancedCorrelationAnalysisService
except ImportError:
    from ...services.advanced_correlation_analysis_service import AdvancedCorrelationAnalysisService

router = APIRouter()

correlation_analysis = AdvancedCorrelationAnalysisService()


@router.post("/correlations/analyze-multivariate")
async def analyze_multivariate_correlations(
    user_id: str = Body(...),
    variables: Dict[str, List[float]] = Body(...)
):
    """Analiza correlaciones multivariadas"""
    try:
        analysis = correlation_analysis.analyze_multivariate_correlations(
            user_id, variables
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando correlaciones: {str(e)}")




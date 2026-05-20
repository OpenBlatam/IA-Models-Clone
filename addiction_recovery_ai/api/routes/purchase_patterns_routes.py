"""
Purchase pattern analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.purchase_pattern_analysis_service import PurchasePatternAnalysisService
except ImportError:
    from ...services.purchase_pattern_analysis_service import PurchasePatternAnalysisService

router = APIRouter()

purchase_analysis = PurchasePatternAnalysisService()


@router.post("/purchases/record")
async def record_purchase(
    user_id: str = Body(...),
    purchase_data: Dict = Body(...)
):
    """Registra una compra"""
    try:
        purchase = purchase_analysis.record_purchase(user_id, purchase_data)
        return JSONResponse(content=purchase)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando compra: {str(e)}")


@router.post("/purchases/analyze-patterns")
async def analyze_purchase_patterns(
    user_id: str = Body(...),
    purchases: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de compra"""
    try:
        analysis = purchase_analysis.analyze_purchase_patterns(
            user_id, purchases, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")




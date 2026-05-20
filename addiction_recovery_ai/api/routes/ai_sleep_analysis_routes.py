"""
AI sleep analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.ai_sleep_analysis_service import AISleepAnalysisService
except ImportError:
    from ...services.ai_sleep_analysis_service import AISleepAnalysisService

router = APIRouter()

ai_sleep = AISleepAnalysisService()


@router.post("/ai-sleep/analyze")
async def analyze_sleep_with_ai(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...)
):
    """Analiza sueño con IA"""
    try:
        analysis = ai_sleep.analyze_sleep_with_ai(user_id, sleep_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando sueño: {str(e)}")




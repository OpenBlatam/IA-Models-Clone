"""
Advanced emotion tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_emotion_tracking_service import AdvancedEmotionTrackingService
except ImportError:
    from ...services.advanced_emotion_tracking_service import AdvancedEmotionTrackingService

router = APIRouter()

advanced_emotion = AdvancedEmotionTrackingService()


@router.post("/emotions/record")
async def record_emotion(
    user_id: str = Body(...),
    emotion_data: Dict = Body(...)
):
    """Registra emoción"""
    try:
        emotion = advanced_emotion.record_emotion(user_id, emotion_data)
        return JSONResponse(content=emotion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando emoción: {str(e)}")


@router.post("/emotions/analyze-patterns")
async def analyze_emotion_patterns(
    user_id: str = Body(...),
    emotions: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones emocionales"""
    try:
        analysis = advanced_emotion.analyze_emotion_patterns(user_id, emotions, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")




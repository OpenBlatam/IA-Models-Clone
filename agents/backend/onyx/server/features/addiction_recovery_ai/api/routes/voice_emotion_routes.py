"""
Voice emotion recognition routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.voice_emotion_recognition_service import VoiceEmotionRecognitionService
except ImportError:
    from ...services.voice_emotion_recognition_service import VoiceEmotionRecognitionService

router = APIRouter()

voice_emotion = VoiceEmotionRecognitionService()


@router.post("/voice-emotion/analyze")
async def analyze_voice_emotions(
    user_id: str = Body(...),
    audio_data: str = Body(...),
    metadata: Optional[Dict] = Body(None)
):
    """Analiza emociones en la voz"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        analysis = voice_emotion.analyze_voice_emotions(user_id, audio_bytes, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando emociones: {str(e)}")


@router.post("/voice-emotion/detect-risk")
async def detect_emotional_risk_from_voice(
    user_id: str = Body(...),
    audio_data: str = Body(...)
):
    """Detecta riesgo emocional desde voz"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        risk_analysis = voice_emotion.detect_emotional_risk_from_voice(user_id, audio_bytes)
        return JSONResponse(content=risk_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando riesgo: {str(e)}")




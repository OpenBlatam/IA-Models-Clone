"""
Voice analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.voice_analysis_service import VoiceAnalysisService
except ImportError:
    from ...services.voice_analysis_service import VoiceAnalysisService

router = APIRouter()

voice_analysis = VoiceAnalysisService()


@router.post("/voice/analyze")
async def analyze_voice(
    user_id: str = Body(...),
    duration_seconds: float = Body(...),
    metadata: Optional[Dict] = Body(None)
):
    """Analiza una grabación de voz"""
    try:
        audio_data = b""
        analysis = voice_analysis.analyze_voice_recording(user_id, audio_data, duration_seconds, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando voz: {str(e)}")




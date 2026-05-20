"""
Advanced voice analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.advanced_voice_analysis_service import AdvancedVoiceAnalysisService
except ImportError:
    from ...services.advanced_voice_analysis_service import AdvancedVoiceAnalysisService

router = APIRouter()

advanced_voice = AdvancedVoiceAnalysisService()


@router.post("/voice/advanced/analyze")
async def analyze_voice_advanced(
    user_id: str = Body(...),
    audio_data: str = Body(...),
    metadata: Optional[Dict] = Body(None)
):
    """Analiza grabación de voz avanzada"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        analysis = advanced_voice.analyze_voice_recording(user_id, audio_bytes, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando voz: {str(e)}")


@router.post("/voice/advanced/detect-stress")
async def detect_voice_stress(
    user_id: str = Body(...),
    audio_data: str = Body(...)
):
    """Detecta estrés en la voz"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        stress_analysis = advanced_voice.detect_voice_stress(user_id, audio_bytes)
        return JSONResponse(content=stress_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando estrés: {str(e)}")




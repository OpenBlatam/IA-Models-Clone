"""
Image emotion analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.image_emotion_analysis_service import ImageEmotionAnalysisService
except ImportError:
    from ...services.image_emotion_analysis_service import ImageEmotionAnalysisService

router = APIRouter()

image_emotion = ImageEmotionAnalysisService()


@router.post("/image/analyze-emotions")
async def analyze_image_emotions(
    user_id: str = Body(...),
    image_data: str = Body(...),
    metadata: Optional[Dict] = Body(None)
):
    """Analiza emociones en una imagen"""
    try:
        import base64
        image_bytes = base64.b64decode(image_data)
        
        analysis = image_emotion.analyze_image_emotions(user_id, image_bytes, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando imagen: {str(e)}")




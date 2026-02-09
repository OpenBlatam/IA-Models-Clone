"""
Social media analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.social_media_analysis_service import SocialMediaAnalysisService
except ImportError:
    from ...services.social_media_analysis_service import SocialMediaAnalysisService

router = APIRouter()

social_media_analysis = SocialMediaAnalysisService()


@router.post("/social-media/analyze")
async def analyze_social_activity(
    user_id: str = Body(...),
    social_posts: List[Dict] = Body(...),
    platform: str = Body("general")
):
    """Analiza actividad en redes sociales"""
    try:
        analysis = social_media_analysis.analyze_social_activity(
            user_id, social_posts, platform
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando actividad: {str(e)}")


@router.post("/social-media/detect-triggers")
async def detect_social_triggers(
    user_id: str = Body(...),
    social_content: List[Dict] = Body(...)
):
    """Detecta triggers en contenido social"""
    try:
        triggers = social_media_analysis.detect_social_triggers(user_id, social_content)
        return JSONResponse(content=triggers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando triggers: {str(e)}")




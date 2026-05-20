"""
Advanced social support routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.advanced_social_support_service import AdvancedSocialSupportService
except ImportError:
    from ...services.advanced_social_support_service import AdvancedSocialSupportService

router = APIRouter()

social_support = AdvancedSocialSupportService()


@router.post("/social-support/assess")
async def assess_social_support(
    user_id: str = Body(...),
    support_data: Dict = Body(...)
):
    """Evalúa apoyo social"""
    try:
        assessment = social_support.assess_social_support(user_id, support_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando apoyo: {str(e)}")


@router.post("/social-support/recommend-resources")
async def recommend_support_resources(
    user_id: str = Body(...),
    user_profile: Dict = Body(...)
):
    """Recomienda recursos de apoyo"""
    try:
        resources = social_support.recommend_support_resources(user_id, user_profile)
        return JSONResponse(content=resources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recomendando recursos: {str(e)}")




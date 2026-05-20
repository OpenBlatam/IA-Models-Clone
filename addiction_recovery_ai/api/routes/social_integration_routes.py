"""
Social integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.social_integration_service import SocialIntegrationService
except ImportError:
    from ...services.social_integration_service import SocialIntegrationService

router = APIRouter()

social_integration = SocialIntegrationService()


@router.post("/social/connect")
async def connect_social_account(
    user_id: str = Body(...),
    platform: str = Body(...),
    access_token: str = Body(...)
):
    """Conecta cuenta de red social"""
    try:
        connection = social_integration.connect_social_account(user_id, platform, access_token)
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando cuenta: {str(e)}")


@router.post("/social/share-milestone")
async def share_milestone(
    user_id: str = Body(...),
    milestone_data: Dict = Body(...),
    platforms: List[str] = Body(...)
):
    """Comparte hito en redes sociales"""
    try:
        share_result = social_integration.share_milestone(user_id, milestone_data, platforms)
        return JSONResponse(content=share_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error compartiendo hito: {str(e)}")




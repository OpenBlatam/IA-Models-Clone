"""
Group therapy integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.group_therapy_integration_service import GroupTherapyIntegrationService
except ImportError:
    from ...services.group_therapy_integration_service import GroupTherapyIntegrationService

router = APIRouter()

group_therapy = GroupTherapyIntegrationService()


@router.post("/group-therapy/find-groups")
async def find_suitable_groups(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    preferences: Dict = Body(...)
):
    """Encuentra grupos adecuados"""
    try:
        groups = group_therapy.find_suitable_groups(
            user_id, user_profile, preferences
        )
        return JSONResponse(content=groups)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encontrando grupos: {str(e)}")


@router.post("/group-therapy/track-participation")
async def track_group_participation(
    user_id: str = Body(...),
    group_id: str = Body(...),
    sessions: List[Dict] = Body(...)
):
    """Rastrea participación en grupo"""
    try:
        participation = group_therapy.track_group_participation(
            user_id, group_id, sessions
        )
        return JSONResponse(content=participation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rastreando participación: {str(e)}")




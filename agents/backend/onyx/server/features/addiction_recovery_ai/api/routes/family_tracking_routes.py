"""
Family tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.family_tracking_service import FamilyTrackingService
except ImportError:
    from ...services.family_tracking_service import FamilyTrackingService

router = APIRouter()

family_tracking = FamilyTrackingService()


@router.post("/family/add-member")
async def add_family_member(
    user_id: str = Body(...),
    family_member_name: str = Body(...),
    relationship: str = Body(...),
    email: Optional[str] = Body(None),
    can_view_progress: bool = Body(True)
):
    """Agrega un miembro de la familia"""
    try:
        member = family_tracking.add_family_member(
            user_id, family_member_name, relationship, email, None, can_view_progress
        )
        return JSONResponse(content=member)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando familiar: {str(e)}")


@router.get("/family/dashboard/{user_id}")
async def get_family_dashboard(
    user_id: str,
    family_member_id: Optional[str] = Query(None)
):
    """Obtiene dashboard para familiares"""
    try:
        dashboard = family_tracking.get_family_dashboard(user_id, family_member_id)
        return JSONResponse(content=dashboard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo dashboard: {str(e)}")




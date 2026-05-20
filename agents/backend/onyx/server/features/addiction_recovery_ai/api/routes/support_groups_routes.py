"""
Advanced support groups routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.advanced_support_groups_service import AdvancedSupportGroupsService
except ImportError:
    from ...services.advanced_support_groups_service import AdvancedSupportGroupsService

router = APIRouter()

advanced_support_groups = AdvancedSupportGroupsService()


@router.post("/support-groups/create")
async def create_support_group(
    creator_id: str = Body(...),
    name: str = Body(...),
    description: str = Body(...),
    group_type: str = Body("public")
):
    """Crea un grupo de apoyo"""
    try:
        group = advanced_support_groups.create_support_group(
            creator_id, name, description, group_type
        )
        return JSONResponse(content=group)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando grupo: {str(e)}")


@router.get("/support-groups/search")
async def search_support_groups(
    query: Optional[str] = Query(None),
    addiction_type: Optional[str] = Query(None)
):
    """Busca grupos de apoyo"""
    try:
        groups = advanced_support_groups.search_groups(query, addiction_type)
        return JSONResponse(content={
            "groups": groups,
            "total": len(groups),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando grupos: {str(e)}")




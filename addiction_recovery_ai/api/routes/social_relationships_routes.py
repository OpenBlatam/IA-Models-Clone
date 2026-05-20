"""
Social relationships routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.social_relationships_service import SocialRelationshipsService
except ImportError:
    from ...services.social_relationships_service import SocialRelationshipsService

router = APIRouter()

social_relationships = SocialRelationshipsService()


@router.post("/relationships/add")
async def add_relationship(
    user_id: str = Body(...),
    contact_name: str = Body(...),
    relationship_type: str = Body(...),
    contact_info: Dict = Body(...)
):
    """Agrega una relación"""
    try:
        relationship = social_relationships.add_relationship(
            user_id, contact_name, relationship_type, contact_info
        )
        return JSONResponse(content=relationship)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando relación: {str(e)}")


@router.get("/relationships/network/{user_id}")
async def get_support_network(user_id: str):
    """Obtiene red de apoyo del usuario"""
    try:
        network = social_relationships.get_support_network(user_id)
        return JSONResponse(content=network)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo red: {str(e)}")




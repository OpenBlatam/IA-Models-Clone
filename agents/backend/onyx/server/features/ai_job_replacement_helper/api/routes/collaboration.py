"""
Collaboration endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.collaboration import CollaborationService, CollaborationType

router = APIRouter()
collaboration_service = CollaborationService()


@router.post("/create/{creator_id}")
async def create_collaboration(
    creator_id: str,
    collaboration_type: str,
    title: str,
    description: str,
    goals: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Crear colaboración"""
    try:
        type_enum = CollaborationType(collaboration_type)
        collaboration = collaboration_service.create_collaboration(
            creator_id, type_enum, title, description, goals
        )
        
        return {
            "id": collaboration.id,
            "title": collaboration.title,
            "type": collaboration.collaboration_type.value,
            "members_count": len(collaboration.members),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/join/{collaboration_id}/{user_id}")
async def join_collaboration(collaboration_id: str, user_id: str) -> Dict[str, Any]:
    """Unirse a colaboración"""
    try:
        success = collaboration_service.join_collaboration(collaboration_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Collaboration not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_collaborations(user_id: str) -> Dict[str, Any]:
    """Obtener colaboraciones del usuario"""
    try:
        collaborations = collaboration_service.get_user_collaborations(user_id)
        return {
            "collaborations": [
                {
                    "id": c.id,
                    "title": c.title,
                    "type": c.collaboration_type.value,
                    "members_count": len(c.members),
                    "status": c.status.value,
                }
                for c in collaborations
            ],
            "total": len(collaborations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





"""
Resources endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.resources import ResourcesService, ResourceType, ResourceCategory

router = APIRouter()
resources_service = ResourcesService()


@router.get("/")
async def get_resources(
    resource_type: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Obtener recursos"""
    try:
        type_enum = ResourceType(resource_type) if resource_type else None
        category_enum = ResourceCategory(category) if category else None
        tags_list = tags.split(",") if tags else None
        
        resources = resources_service.get_resources(
            type_enum, category_enum, tags_list, limit
        )
        
        return {
            "resources": [
                {
                    "id": r.id,
                    "title": r.title,
                    "description": r.description,
                    "type": r.resource_type.value,
                    "category": r.category.value,
                    "url": r.url,
                    "rating": r.rating,
                    "views": r.views,
                }
                for r in resources
            ],
            "total": len(resources),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{resource_id}")
async def get_resource(resource_id: str) -> Dict[str, Any]:
    """Obtener recurso específico"""
    try:
        resource = resources_service.get_resource(resource_id)
        if not resource:
            raise HTTPException(status_code=404, detail="Resource not found")
        
        return {
            "id": resource.id,
            "title": resource.title,
            "description": resource.description,
            "type": resource.resource_type.value,
            "category": resource.category.value,
            "url": resource.url,
            "rating": resource.rating,
            "views": resource.views,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bookmark/{user_id}/{resource_id}")
async def bookmark_resource(user_id: str, resource_id: str) -> Dict[str, Any]:
    """Guardar recurso en favoritos"""
    try:
        success = resources_service.bookmark_resource(user_id, resource_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookmarks/{user_id}")
async def get_bookmarks(user_id: str) -> Dict[str, Any]:
    """Obtener recursos guardados"""
    try:
        bookmarks = resources_service.get_user_bookmarks(user_id)
        return {
            "resources": [
                {
                    "id": r.id,
                    "title": r.title,
                    "type": r.resource_type.value,
                }
                for r in bookmarks
            ],
            "total": len(bookmarks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/featured")
async def get_featured(limit: int = 10) -> Dict[str, Any]:
    """Obtener recursos destacados"""
    try:
        featured = resources_service.get_featured_resources(limit)
        return {
            "resources": [
                {
                    "id": r.id,
                    "title": r.title,
                    "rating": r.rating,
                }
                for r in featured
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





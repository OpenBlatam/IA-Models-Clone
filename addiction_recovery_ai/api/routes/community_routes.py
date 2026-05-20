"""
Community routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.community_service import CommunityService
except ImportError:
    from ...services.community_service import CommunityService

router = APIRouter()

community = CommunityService()


@router.post("/community/post")
async def create_community_post(
    user_id: str = Body(...),
    post_type: str = Body(...),
    title: str = Body(...),
    content: str = Body(...),
    is_anonymous: bool = Body(False)
):
    """Crea una publicación en la comunidad"""
    try:
        post = community.create_post(user_id, post_type, title, content, is_anonymous)
        return JSONResponse(content=post)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando publicación: {str(e)}")


@router.get("/community/posts")
async def get_community_posts(
    post_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Obtiene publicaciones de la comunidad"""
    try:
        posts = community.get_community_posts(post_type, limit, offset)
        return JSONResponse(content={
            "posts": posts,
            "total": len(posts),
            "limit": limit,
            "offset": offset,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo publicaciones: {str(e)}")




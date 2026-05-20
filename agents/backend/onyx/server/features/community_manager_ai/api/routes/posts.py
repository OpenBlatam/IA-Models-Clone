"""
Posts API Routes
================

Endpoints para gestión de posts.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(prefix="/posts", tags=["posts"])


class PostCreate(BaseModel):
    content: str
    platforms: List[str]
    scheduled_time: Optional[datetime] = None
    media_paths: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class PostResponse(BaseModel):
    post_id: str
    status: str
    scheduled_time: str
    platforms: List[str]


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


@router.post("/", response_model=PostResponse)
async def create_post(
    post: PostCreate,
    manager = Depends(get_community_manager)
):
    """Crear y programar un nuevo post"""
    try:
        result = manager.schedule_post(
            content=post.content,
            platforms=post.platforms,
            scheduled_time=post.scheduled_time,
            media_paths=post.media_paths,
            tags=post.tags
        )
        return PostResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[dict])
async def get_posts(
    status: Optional[str] = None,
    manager = Depends(get_community_manager)
):
    """Obtener todos los posts"""
    try:
        posts = manager.scheduler.get_all_posts(status=status)
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{post_id}", response_model=dict)
async def get_post(
    post_id: str,
    manager = Depends(get_community_manager)
):
    """Obtener un post específico"""
    try:
        post = manager.scheduler.get_post(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post no encontrado")
        return post
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{post_id}/publish")
async def publish_post(
    post_id: str,
    manager = Depends(get_community_manager)
):
    """Publicar un post inmediatamente"""
    try:
        post = manager.scheduler.get_post(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post no encontrado")
        
        results = manager.publish_now(
            content=post.get("content"),
            platforms=post.get("platforms", []),
            media_paths=post.get("media_paths")
        )
        
        manager.scheduler.mark_as_published(post_id, results)
        
        return {"status": "published", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{post_id}")
async def cancel_post(
    post_id: str,
    manager = Depends(get_community_manager)
):
    """Cancelar un post programado"""
    try:
        success = manager.scheduler.cancel_post(post_id)
        if not success:
            raise HTTPException(status_code=404, detail="Post no encontrado")
        return {"status": "cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





"""
Batch API Routes
================

Endpoints para operaciones por lotes.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/batch", tags=["batch"])


class BatchPostCreate(BaseModel):
    content: str
    platforms: List[str]
    scheduled_time: Optional[str] = None
    media_paths: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class BatchPublishRequest(BaseModel):
    post_ids: List[str]


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


def get_batch_service():
    """Dependency para obtener BatchService"""
    from ...services.batch_service import BatchService
    return BatchService()


@router.post("/schedule", response_model=dict)
async def schedule_batch(
    posts: List[BatchPostCreate],
    manager = Depends(get_community_manager),
    batch_service = Depends(get_batch_service)
):
    """Programar múltiples posts"""
    try:
        from datetime import datetime
        
        posts_data = []
        for post in posts:
            scheduled_time = None
            if post.scheduled_time:
                scheduled_time = datetime.fromisoformat(post.scheduled_time)
            
            posts_data.append({
                "content": post.content,
                "platforms": post.platforms,
                "scheduled_time": scheduled_time or datetime.now(),
                "media_paths": post.media_paths or [],
                "tags": post.tags or [],
                "created_at": datetime.now()
            })
        
        post_ids = batch_service.schedule_batch(posts_data, manager.scheduler)
        
        return {
            "scheduled": len(post_ids),
            "post_ids": post_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish", response_model=dict)
async def publish_batch(
    request: BatchPublishRequest,
    manager = Depends(get_community_manager),
    batch_service = Depends(get_batch_service)
):
    """Publicar múltiples posts"""
    try:
        # Obtener posts
        posts = []
        for post_id in request.post_ids:
            post = manager.scheduler.get_post(post_id)
            if post:
                posts.append(post)
        
        if not posts:
            raise HTTPException(status_code=404, detail="No se encontraron posts")
        
        # Publicar en lote
        results = batch_service.publish_batch(
            posts=posts,
            connector=manager.social_connector
        )
        
        # Marcar como publicados
        for result in results["results"]:
            if result["status"] == "success":
                manager.scheduler.mark_as_published(
                    result["post_id"],
                    result.get("result", {})
                )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


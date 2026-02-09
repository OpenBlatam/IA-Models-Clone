"""
Community endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.community import CommunityService, PostType

router = APIRouter()
community_service = CommunityService()


@router.post("/posts/{user_id}")
async def create_post(
    user_id: str,
    title: str,
    content: str,
    post_type: str,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Crear un nuevo post"""
    try:
        post_type_enum = PostType(post_type)
        post = community_service.create_post(user_id, title, content, post_type_enum, tags)
        return {
            "id": post.id,
            "title": post.title,
            "content": post.content,
            "type": post.post_type.value,
            "tags": post.tags,
            "created_at": post.created_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/posts")
async def get_posts(
    post_type: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """Obtener posts"""
    try:
        post_type_enum = PostType(post_type) if post_type else None
        tags_list = tags.split(",") if tags else None
        posts = community_service.get_posts(post_type_enum, tags_list, limit, offset)
        return {
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "content": p.content[:200] + "..." if len(p.content) > 200 else p.content,
                    "type": p.post_type.value,
                    "tags": p.tags,
                    "likes": p.likes,
                    "views": p.views,
                    "comments_count": p.comments_count,
                    "created_at": p.created_at.isoformat(),
                }
                for p in posts
            ],
            "total": len(posts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/posts/{post_id}")
async def get_post(post_id: str) -> Dict[str, Any]:
    """Obtener un post específico"""
    try:
        post = community_service.get_post(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return {
            "id": post.id,
            "title": post.title,
            "content": post.content,
            "type": post.post_type.value,
            "tags": post.tags,
            "likes": post.likes,
            "views": post.views,
            "comments_count": post.comments_count,
            "created_at": post.created_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/posts/{post_id}/like/{user_id}")
async def like_post(post_id: str, user_id: str) -> Dict[str, Any]:
    """Dar like a un post"""
    try:
        liked = community_service.like_post(post_id, user_id)
        post = community_service.get_post(post_id)
        return {
            "liked": liked,
            "total_likes": post.likes if post else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/posts/{post_id}/comments/{user_id}")
async def create_comment(
    post_id: str,
    user_id: str,
    content: str,
    parent_comment_id: Optional[str] = None
) -> Dict[str, Any]:
    """Crear comentario"""
    try:
        comment = community_service.create_comment(post_id, user_id, content, parent_comment_id)
        return {
            "id": comment.id,
            "content": comment.content,
            "created_at": comment.created_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/posts/{post_id}/comments")
async def get_comments(post_id: str) -> Dict[str, Any]:
    """Obtener comentarios de un post"""
    try:
        comments = community_service.get_comments(post_id)
        return {
            "comments": [
                {
                    "id": c.id,
                    "content": c.content,
                    "likes": c.likes,
                    "created_at": c.created_at.isoformat(),
                }
                for c in comments
            ],
            "total": len(comments),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
async def get_trending(limit: int = 10) -> Dict[str, Any]:
    """Obtener posts trending"""
    try:
        posts = community_service.get_trending_posts(limit)
        return {
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "likes": p.likes,
                    "views": p.views,
                    "comments_count": p.comments_count,
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_posts(query: str, limit: int = 20) -> Dict[str, Any]:
    """Buscar posts"""
    try:
        posts = community_service.search_posts(query, limit)
        return {
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "type": p.post_type.value,
                }
                for p in posts
            ],
            "total": len(posts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from .models import BlogPost
from pydantic import BaseModel, Field
from .utils import find_post_index, not_found_error, conflict_error, bad_request_error

from typing import Any, List, Dict, Optional
import logging
router = APIRouter()

# Simulación de base de datos en memoria
posts_db: List[BlogPost] = []

# --- Modelos de request/response ---
class BlogPostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: List[str] = []

class BlogPostUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)
    content: str | None = Field(None, min_length=1)
    tags: List[str] | None = None
    is_published: bool | None = None

# --- Dependencia para acceso a datos asíncrono ---
async def get_posts_db() -> List[BlogPost]:
    await asyncio.sleep(0.01)
    return posts_db

# --- Rutas ---
@router.get("/", response_model=List[BlogPost])
async def list_posts(posts: List[BlogPost] = Depends(get_posts_db)) -> List[BlogPost]:
    return posts

@router.get("/{post_id}", response_model=BlogPost)
async def get_post(post_id: int, posts: List[BlogPost] = Depends(get_posts_db)) -> BlogPost:
    idx = find_post_index(posts, post_id)
    if idx == -1:
        raise not_found_error("BlogPost")
    return posts[idx]

@router.post("/", response_model=BlogPost, status_code=status.HTTP_201_CREATED)
async def create_post(post: BlogPostCreate, posts: List[BlogPost] = Depends(get_posts_db)) -> BlogPost:
    if any(p.title == post.title for p in posts):
        raise conflict_error("BlogPost")
    new_post = BlogPost(id=len(posts) + 1, **post.dict())
    posts.append(new_post)
    return new_post

@router.patch("/{post_id}", response_model=BlogPost)
async def update_post(post_id: int, post: BlogPostUpdate, posts: List[BlogPost] = Depends(get_posts_db)) -> BlogPost:
    idx = find_post_index(posts, post_id)
    if idx == -1:
        raise not_found_error("BlogPost")
    stored_post = posts[idx]
    update_data = post.dict(exclude_unset=True)
    updated_post = stored_post.copy(update=update_data)
    posts[idx] = updated_post
    return updated_post

@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(post_id: int, posts: List[BlogPost] = Depends(get_posts_db)):
    idx = find_post_index(posts, post_id)
    if idx == -1:
        raise not_found_error("BlogPost")
    posts.pop(idx) 
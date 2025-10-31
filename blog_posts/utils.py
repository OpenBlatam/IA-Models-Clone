from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
from fastapi import HTTPException
from typing import List
from .models import BlogPost

from typing import Any, List, Dict, Optional
import asyncio
logger = logging.getLogger("blog_system")

def find_post_index(posts: List[BlogPost], post_id: int) -> int:
    for idx, post in enumerate(posts):
        if post.id == post_id:
            return idx
    return -1

def not_found_error(entity: str = "Resource") -> HTTPException:
    logger.error(f"{entity} not found.")
    return HTTPException(status_code=404, detail=f"{entity} not found.")

def conflict_error(entity: str = "Resource") -> HTTPException:
    logger.error(f"{entity} already exists.")
    return HTTPException(status_code=400, detail=f"{entity} already exists.")

async def bad_request_error(message: str) -> HTTPException:
    logger.error(message)
    return HTTPException(status_code=400, detail=message) 
"""
API dependencies for dependency injection
"""

from typing import Annotated, AsyncGenerator
from fastapi import Depends

from ..config.database import get_db_session
from ..services.blog_service import BlogService
from ..services.user_service import UserService
from ..services.comment_service import CommentService
from ..core.security import get_current_user, get_current_active_user
from ..core.caching import CacheService
from ..config.settings import get_settings
from sqlalchemy.ext.asyncio import AsyncSession


# Database dependencies
async def get_blog_service(
    session: AsyncSession = Depends(get_db_session)
) -> BlogService:
    """Get blog service instance."""
    return BlogService(session)


async def get_user_service(
    session: AsyncSession = Depends(get_db_session)
) -> UserService:
    """Get user service instance."""
    return UserService(session)


async def get_comment_service(
    session: AsyncSession = Depends(get_db_session)
) -> CommentService:
    """Get comment service instance."""
    return CommentService(session)


# Cache dependencies
async def get_cache_service() -> CacheService:
    """Get cache service instance."""
    settings = get_settings()
    return CacheService(settings.redis_url)


# Type aliases for cleaner code
BlogServiceDep = Annotated[BlogService, Depends(get_blog_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
CommentServiceDep = Annotated[CommentService, Depends(get_comment_service)]
CacheServiceDep = Annotated[CacheService, Depends(get_cache_service)]
CurrentUserDep = Annotated[dict, Depends(get_current_active_user)]































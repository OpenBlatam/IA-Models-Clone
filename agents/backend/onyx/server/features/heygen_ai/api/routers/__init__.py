from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .video_routes import router as video_router
from .health_routes import router as health_router
from .user_routes import router as user_router
from .auth_routes import router as auth_router
from .admin_routes import router as admin_router
from .analytics_routes import router as analytics_router
from fastapi import APIRouter
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Routers module for HeyGen AI API
Exports all route modules and main router.
"""


# Main router that combines all sub-routes

main_router = APIRouter()

# Include all sub-routes
main_router.include_router(video_router, prefix="/videos", tags=["videos"])
main_router.include_router(health_router, prefix="/health", tags=["health"])
main_router.include_router(user_router, prefix="/users", tags=["users"])
main_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
main_router.include_router(admin_router, prefix="/admin", tags=["admin"])
main_router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])

# Named exports
__all__ = [
    "main_router",
    "video_router",
    "health_router",
    "user_router", 
    "auth_router",
    "admin_router",
    "analytics_router"
] 
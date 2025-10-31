from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import structlog
from .routers.users import router as users_router
from .routers.videos import router as videos_router
from .routers.auth import router as auth_router
from .routers.health import router as health_router
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API Router

Main API router that combines all endpoint routers.
"""



logger = structlog.get_logger()


async def create_api_router() -> APIRouter:
    """
    Create and configure the main API router.
    
    Combines all sub-routers with proper prefixes and tags.
    """
    api_router = APIRouter()
    
    # Authentication routes
    api_router.include_router(
        auth_router,
        prefix="/auth",
        tags=["authentication"]
    )
    
    # User management routes
    api_router.include_router(
        users_router,
        prefix="/users", 
        tags=["users"]
    )
    
    # Video processing routes
    api_router.include_router(
        videos_router,
        prefix="/videos",
        tags=["videos"]
    )
    
    # Health and monitoring routes
    api_router.include_router(
        health_router,
        prefix="/health",
        tags=["health"]
    )
    
    # Root endpoint
    @api_router.get("/", response_model=Dict[str, Any])
    async def root():
        """API root endpoint with basic information."""
        return {
            "message": "HeyGen AI FastAPI - Clean Architecture",
            "version": "2.0.0",
            "status": "running",
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
            "architecture": "clean_architecture",
            "features": [
                "user_management",
                "video_processing", 
                "ai_generation",
                "real_time_status",
                "comprehensive_monitoring"
            ]
        }
    
    return api_router 
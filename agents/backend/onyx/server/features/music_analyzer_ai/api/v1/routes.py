"""
API v1 Routes

Aggregates all v1 controllers into a single router.
"""

from fastapi import APIRouter

from fastapi.middleware.base import BaseHTTPMiddleware

from .controllers.analysis_controller import router as analysis_router
from .controllers.search_controller import router as search_router
from .controllers.recommendations_controller import router as recommendations_router
from .middleware.error_handler import ErrorHandlerMiddleware

# Create main v1 router
v1_router = APIRouter(prefix="/v1/music", tags=["Music API v1"])

# Include all controller routers
v1_router.include_router(analysis_router)
v1_router.include_router(search_router)
v1_router.include_router(recommendations_router)

__all__ = ["v1_router"]


"""
API v3 Package
==============

API v3 routers and endpoints.
"""

from .workflow_router import router as workflow_router
from .analytics_router import router as analytics_router
from .admin_router import router as admin_router
from .auth_router import router as auth_router
from .ai_router import router as ai_router
from .file_router import router as file_router
from .search_router import router as search_router
from .monitoring_router import router as monitoring_router

from fastapi import APIRouter

# Create main v3 router
v3_router = APIRouter(prefix="/api/v3")

# Include all routers
v3_router.include_router(workflow_router, prefix="/workflows", tags=["Workflows"])
v3_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
v3_router.include_router(admin_router, prefix="/admin", tags=["Admin"])
v3_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
v3_router.include_router(ai_router, prefix="/ai", tags=["AI Operations"])
v3_router.include_router(file_router, prefix="/files", tags=["File Operations"])
v3_router.include_router(search_router, prefix="/search", tags=["Search Operations"])
v3_router.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring Operations"])

__all__ = [
    "v3_router",
    "workflow_router",
    "analytics_router",
    "admin_router",
    "auth_router",
    "ai_router",
    "file_router",
    "search_router",
    "monitoring_router"
]
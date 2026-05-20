"""
API Package
===========

API routers and endpoints.
"""

from .v3 import v3_router

from fastapi import APIRouter

# Create main API router
api_router = APIRouter()

# Include v3 router
api_router.include_router(v3_router)

__all__ = [
    "api_router",
    "v3_router"
]
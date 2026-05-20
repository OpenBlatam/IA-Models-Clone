"""
Progress routes module
"""

from fastapi import APIRouter

from .endpoints import router as progress_endpoints_router

router = APIRouter()

router.include_router(progress_endpoints_router)


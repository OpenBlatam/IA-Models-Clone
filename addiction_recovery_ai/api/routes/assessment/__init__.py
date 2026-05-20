"""
Assessment routes module
"""

from fastapi import APIRouter

from .endpoints import router as assessment_endpoints_router

router = APIRouter()

router.include_router(assessment_endpoints_router)


"""
Rutas principales de la API v1
"""

from fastapi import APIRouter
from .controllers.extract_controller import router as extract_router
from .controllers.metrics_controller import router as metrics_router

router = APIRouter(prefix="/api/v1")

router.include_router(extract_router)
router.include_router(metrics_router)


"""
Router principal para la API de comunidad Lovable (optimizado)

Incluye todos los routers de la aplicación.
"""

import logging
from fastapi import APIRouter

from .routes import router as community_router
from .health import router as health_router
from .metrics import router as metrics_router
from .ai_routes import router as ai_router
from .root import router as root_router

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/lovable",
    tags=["lovable-community"]
)

# Incluir routers
router.include_router(root_router)
router.include_router(community_router)
router.include_router(health_router)
router.include_router(metrics_router)
router.include_router(ai_router)

logger.info("Lovable Community API router initialized")

__all__ = ["router"]


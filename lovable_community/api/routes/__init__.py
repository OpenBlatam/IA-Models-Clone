"""
Rutas API modulares para la comunidad Lovable

Combina todos los routers modulares en un solo router principal.
"""

from fastapi import APIRouter

from .analytics import router as analytics_router
from .bulk import router as bulk_router
from .chats import router as chats_router
from .remixes import router as remixes_router
from .search import router as search_router
from .stats import router as stats_router
from .votes import router as votes_router

router = APIRouter(
    prefix="/community",
    tags=["community"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        401: {"description": "Unauthorized - Autenticación requerida"},
        404: {"description": "Not found - Recurso no encontrado"},
        422: {"description": "Unprocessable entity - Error de validación"},
        500: {"description": "Internal server error - Error del servidor"}
    }
)

router.include_router(analytics_router)
router.include_router(bulk_router)
router.include_router(chats_router)
router.include_router(remixes_router)
router.include_router(search_router)
router.include_router(stats_router)
router.include_router(votes_router)

__all__ = ["router"]


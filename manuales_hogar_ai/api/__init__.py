"""API del módulo."""

from fastapi import APIRouter
from .routes import (
    manuales_router,
    history_router,
    ratings_router,
    export_router,
    search_router,
    share_router,
    notifications_router,
    templates_router,
    analytics_router,
    ml_router,
    streaming_router,
    health_router
)

router = APIRouter()
router.include_router(manuales_router, prefix="/manuales", tags=["manuales"])
router.include_router(history_router)
router.include_router(ratings_router)
router.include_router(export_router)
router.include_router(search_router)
router.include_router(share_router)
router.include_router(notifications_router)
router.include_router(templates_router)
router.include_router(analytics_router)
router.include_router(ml_router)
router.include_router(streaming_router)
router.include_router(health_router)

__all__ = [
    "router",
    "manuales_router",
    "history_router",
    "ratings_router",
    "export_router",
    "search_router",
    "share_router",
    "notifications_router",
    "templates_router",
    "analytics_router",
    "ml_router",
    "streaming_router",
    "health_router"
]


"""Rutas de la API."""

from .manuales import router as manuales_router
from .history import router as history_router
from .ratings import router as ratings_router
from .export import router as export_router
from .search import router as search_router
from .share import router as share_router
from .notifications import router as notifications_router
from .templates import router as templates_router
from .analytics import router as analytics_router
from .ml import router as ml_router
from .streaming import router as streaming_router
from .health import router as health_router
from .metrics import router as metrics_router

__all__ = [
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
    "health_router",
    "metrics_router",
]


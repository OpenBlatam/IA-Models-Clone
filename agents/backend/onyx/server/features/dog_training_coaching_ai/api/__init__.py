"""API module."""

from .routes import router
from .routes.metrics import router as metrics_router

__all__ = ["router", "metrics_router"]


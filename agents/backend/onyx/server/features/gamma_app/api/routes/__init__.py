"""
API Routes
Modular route definitions organized by domain
"""

from .content import router as content_router
from .collaboration import router as collaboration_router
from .export import router as export_router
from .analytics import router as analytics_router

__all__ = [
    "content_router",
    "collaboration_router",
    "export_router",
    "analytics_router"
]








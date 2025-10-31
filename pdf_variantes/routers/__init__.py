"""
PDF Variantes Routers
=====================

FastAPI routers for PDF variantes features following best practices.
"""

from .pdf_router import router as pdf_router
from .analytics_router import router as analytics_router
from .collaboration_router import router as collaboration_router

__all__ = ["pdf_router", "analytics_router", "collaboration_router"]

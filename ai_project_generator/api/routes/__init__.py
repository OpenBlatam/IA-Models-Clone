"""
API Routes - Endpoints organizados por dominio
==============================================

Endpoints organizados por dominio siguiendo principios de microservicios.
"""

from .projects import router as projects_router
from .generation import router as generation_router
from .validation import router as validation_router
from .export import router as export_router
from .deployment import router as deployment_router
from .analytics import router as analytics_router
from .health import router as health_router

__all__ = [
    "projects_router",
    "generation_router",
    "validation_router",
    "export_router",
    "deployment_router",
    "analytics_router",
    "health_router",
]


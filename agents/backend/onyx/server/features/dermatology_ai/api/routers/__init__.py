"""
Modular routers for Dermatology AI API
"""

from .auth_router import router as auth_router
from .analysis_router import router as analysis_router
from .recommendations_router import router as recommendations_router
from .tracking_router import router as tracking_router
from .products_router import router as products_router
from .ml_router import router as ml_router
from .integrations_router import router as integrations_router
from .reports_router import router as reports_router
from .social_router import router as social_router
from .health_router import router as health_router
from .performance_router import router as performance_router
from .router_manager import RouterManager, get_router_manager
from .base_router import create_base_router, handle_api_error, create_success_response

__all__ = [
    "auth_router",
    "analysis_router",
    "recommendations_router",
    "tracking_router",
    "products_router",
    "ml_router",
    "integrations_router",
    "reports_router",
    "social_router",
    "health_router",
    "performance_router",
    "RouterManager",
    "get_router_manager",
    "create_base_router",
    "handle_api_error",
    "create_success_response",
]


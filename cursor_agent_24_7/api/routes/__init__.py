"""
API Routes - Rutas de la API
=============================

Módulo que contiene todas las rutas HTTP de la API.
"""

from .agent_routes import router as agent_router
from .task_routes import router as task_router
from .notification_routes import router as notification_router

# Search routes (opcional)
try:
    from .search_routes import router as search_router
except ImportError:
    search_router = None

# Perplexity routes (opcional)
try:
    from .perplexity_routes import router as perplexity_router
except ImportError:
    perplexity_router = None

__all__ = ["agent_router", "task_router", "notification_router"]
if search_router:
    __all__.append("search_router")
if perplexity_router:
    __all__.append("perplexity_router")


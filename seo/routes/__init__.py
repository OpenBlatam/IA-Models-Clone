from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .analysis import router as analysis_router
from .database import router as database_router
from .external_api import router as external_api_router
from .persistence import router as persistence_router
from .performance import router as performance_router
from .cache import router as cache_router
from .health import router as health_router
from .data import router as data_router
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Routes package for Ultra-Optimized SEO Service v15.

This package contains organized route modules for different API functionalities:
- analysis: Core SEO analysis endpoints
- database: Database operation endpoints
- external_api: External API operation endpoints
- persistence: Data persistence endpoints
- performance: Performance monitoring endpoints
- cache: Cache management endpoints
- health: Health check and monitoring endpoints
- data: Data access and lazy loading endpoints
"""


__all__ = [
    "analysis_router",
    "database_router", 
    "external_api_router",
    "persistence_router",
    "performance_router",
    "cache_router",
    "health_router",
    "data_router"
] 
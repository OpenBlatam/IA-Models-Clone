from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter
from typing import List
from .message_routes import router as message_router
from .batch_routes import router as batch_router
from .analysis_routes import router as analysis_router
from .cache_routes import router as cache_router
from .health_routes import router as health_router
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Main router module for Key Messages feature.
Exports all sub-routes and follows modular structure.
"""

# Import sub-routes

# Create main router
router = APIRouter(prefix="/key-messages", tags=["key-messages"])

# Include sub-routes
router.include_router(message_router, prefix="/messages")
router.include_router(batch_router, prefix="/batch")
router.include_router(analysis_router, prefix="/analysis")
router.include_router(cache_router, prefix="/cache")
router.include_router(health_router, prefix="/health")

# Export router
__all__ = ["router"] 
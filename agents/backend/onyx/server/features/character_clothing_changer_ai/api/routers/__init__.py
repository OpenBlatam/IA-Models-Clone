"""
Routers Module
==============

API routers for different endpoints.
"""

from .clothing_router import router as clothing_router
from .tensor_router import router as tensor_router
from .model_router import router as model_router
from .health_router import router as health_router
from .image_router import router as image_router

__all__ = [
    "clothing_router",
    "tensor_router",
    "model_router",
    "health_router",
    "image_router",
]



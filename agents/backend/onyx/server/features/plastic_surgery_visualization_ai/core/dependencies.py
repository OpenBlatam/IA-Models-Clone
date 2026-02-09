"""Dependency injection for services (FastAPI dependencies)."""

from fastapi import Depends
from core.factories import (
    create_cache,
    create_image_processor,
    create_ai_processor,
    create_visualization_service
)
from utils.cache_advanced import Cache
from core.services.ai_processor import AIProcessor
from core.services.image_processor import ImageProcessor
from services.visualization_service import VisualizationService


# FastAPI dependency functions
def get_cache() -> Cache:
    """Get cache instance (FastAPI dependency)."""
    return create_cache()


def get_image_processor() -> ImageProcessor:
    """Get image processor instance (FastAPI dependency)."""
    return create_image_processor()


def get_ai_processor() -> AIProcessor:
    """Get AI processor instance (FastAPI dependency)."""
    return create_ai_processor()


def get_visualization_service() -> VisualizationService:
    """Get visualization service instance (FastAPI dependency)."""
    return create_visualization_service()


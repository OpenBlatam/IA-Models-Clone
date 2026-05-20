"""
Application Lifespan Management
Handles startup and shutdown logic for the FastAPI application
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..application.di.container import get_container, reset_container
from ..core.content_generator import ContentGenerator
from ..services.collaboration_service import CollaborationService
from ..services.analytics_service import AnalyticsService
from ..infrastructure.database.session import get_db_manager
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

async def _startup_application(app: FastAPI) -> None:
    """Perform application startup tasks"""
    settings = get_settings()
    logger.info("Starting Gamma App API...", extra={
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    })
    
    try:
        container = get_container()
        db_manager = get_db_manager()
        
        app.state.container = container
        app.state.db_manager = db_manager
        
        logger.info("Gamma App API started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

async def _shutdown_application(app: FastAPI) -> None:
    """Perform application shutdown tasks"""
    logger.info("Shutting down Gamma App API...")
    
    try:
        container = getattr(app.state, "container", None)
        if container:
            await container.shutdown()
        
        db_manager = getattr(app.state, "db_manager", None)
        if db_manager:
            db_manager.close()
        
        logger.info("Gamma App API shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    await _startup_application(app)
    
    try:
        yield
    finally:
        await _shutdown_application(app)

def _get_service_from_container(service_type: type) -> Optional[object]:
    """Get service instance from DI container"""
    try:
        container = get_container()
        return container.get(service_type)
    except Exception as e:
        logger.warning(f"Error getting {service_type.__name__}: {e}")
        return None

def get_content_generator_instance() -> Optional[ContentGenerator]:
    """Get content generator instance from DI container"""
    return _get_service_from_container(ContentGenerator)

def get_collaboration_service_instance() -> Optional[CollaborationService]:
    """Get collaboration service instance from DI container"""
    return _get_service_from_container(CollaborationService)

def get_analytics_service_instance() -> Optional[AnalyticsService]:
    """Get analytics service instance from DI container"""
    return _get_service_from_container(AnalyticsService)

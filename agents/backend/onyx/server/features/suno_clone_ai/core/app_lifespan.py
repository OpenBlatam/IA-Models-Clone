"""
Application lifespan management for startup and shutdown
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI

from config.settings import settings
from bootstrap import bootstrap_application, initialize_modules, shutdown_modules
from core.performance_optimizer import get_performance_optimizer
from core.fast_cache import get_fast_cache
from core.graceful_degradation import get_graceful_degradation

logger = logging.getLogger(__name__)


async def _initialize_cache():
    """Initialize cache with Redis if available"""
    fast_cache = get_fast_cache()
    if settings.redis_url:
        try:
            import redis.asyncio as redis
            redis_client = await redis.from_url(settings.redis_url)
            fast_cache.set_l2_cache(redis_client)
            logger.info("Fast cache L2 (Redis) initialized")
        except ImportError:
            logger.warning("Redis library not available, skipping L2 cache")
        except Exception as e:
            logger.warning(f"Could not initialize Redis cache: {e}")
    return fast_cache


async def _startup_application(app: FastAPI) -> None:
    """Perform application startup tasks"""
    try:
        perf_optimizer = get_performance_optimizer()
        perf_optimizer.optimize_all()
        logger.info("Performance optimizations applied")
    except Exception as e:
        logger.error(f"Failed to apply performance optimizations: {e}", exc_info=True)
    
    fast_cache = await _initialize_cache()
    
    logger.info("Starting application...", extra={
        "service": settings.app_name,
        "version": settings.app_version,
        "is_lambda": settings.is_lambda
    })
    
    try:
        bootstrap_result = bootstrap_application(app)
        registry = bootstrap_result["registry"]
        
        initialization_success = await initialize_modules(registry)
        if not initialization_success:
            logger.error("Some modules failed to initialize")
        
        app.state.module_registry = registry
        app.state.fast_cache = fast_cache
        
        degradation = get_graceful_degradation()
        app.state.graceful_degradation = degradation
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise


async def _shutdown_application(app: FastAPI) -> None:
    """Perform application shutdown tasks"""
    logger.info("Shutting down...", extra={"service": settings.app_name})
    
    try:
        registry = getattr(app.state, "module_registry", None)
        if registry:
            await shutdown_modules(registry)
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}", exc_info=True)


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """
    Manage application lifecycle: startup and shutdown.
    
    Handles:
    - Performance optimizations
    - Cache initialization (L1 and L2)
    - Module registration and initialization
    - Graceful degradation setup
    """
    await _startup_application(app)
    
    try:
        yield
    finally:
        await _shutdown_application(app)


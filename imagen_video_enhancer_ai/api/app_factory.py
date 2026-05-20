"""
Application Factory
===================

Factory for creating and configuring FastAPI application.
"""

import logging
from typing import Optional
from pathlib import Path
from fastapi import FastAPI

from ..core.application import Application
from ..core.enhancer_agent import EnhancerAgent
from ..config.enhancer_config import EnhancerConfig
from .dependencies import (
    set_agent,
    set_dashboard,
    set_managers
)
from .middleware import setup_cors, rate_limit_middleware

# Import all routes
from .routes import (
    task_routes,
    enhancement_routes,
    service_routes,
    batch_routes,
    webhook_routes,
    auth_routes,
    analysis_routes,
    export_routes,
    notification_routes,
    config_routes,
    monitoring_routes,
    metrics_routes
)

logger = logging.getLogger(__name__)


def create_app(
    config: Optional[EnhancerConfig] = None,
    config_file: Optional[Path] = None,
    title: str = "Imagen Video Enhancer AI API",
    version: str = "1.0.0",
    description: str = "API for image and video enhancement with OpenRouter and TruthGPT"
) -> tuple[FastAPI, Application]:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Optional enhancer config
        config_file: Optional config file path
        title: API title
        version: API version
        description: API description
        
    Returns:
        Tuple of (FastAPI app, Application instance)
    """
    # Create application
    app_instance = Application(config=config, config_file=config_file)
    
    # Create FastAPI app
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Setup middleware
    setup_cors(app)
    app.middleware("http")(rate_limit_middleware)
    
    # Include all routers
    app.include_router(task_routes.router)
    app.include_router(enhancement_routes.router)
    app.include_router(service_routes.router)
    app.include_router(batch_routes.router)
    app.include_router(webhook_routes.router)
    app.include_router(auth_routes.router)
    app.include_router(analysis_routes.router)
    app.include_router(export_routes.router)
    app.include_router(notification_routes.router)
    app.include_router(config_routes.router)
    app.include_router(monitoring_routes.router)
    app.include_router(metrics_routes.router)
    
    # Setup startup/shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        await app_instance.initialize()
        await app_instance.start()
        
        # Set global dependencies
        agent = app_instance.get_agent()
        set_agent(agent)
        
        # Set managers (would be loaded from app_instance)
        from ..core.monitoring_dashboard import MonitoringDashboard
        from ..core.auth import AuthManager
        from ..core.notification_system import NotificationManager
        from ..core.metrics_collector import MetricsCollector
        from ..core.event_bus import EventBus
        from ..core.rate_limiter import RateLimiter, RateLimitConfig
        from ..core.constants import DEFAULT_RATE_LIMIT_RPS, DEFAULT_RATE_LIMIT_BURST
        
        dashboard = MonitoringDashboard(agent)
        auth_manager = AuthManager()
        notification_manager = NotificationManager()
        metrics_collector = MetricsCollector()
        event_bus = EventBus()
        rate_limiter = RateLimiter(
            default_config=RateLimitConfig(
                requests_per_second=DEFAULT_RATE_LIMIT_RPS,
                burst_size=DEFAULT_RATE_LIMIT_BURST
            )
        )
        
        set_dashboard(dashboard)
        set_managers(
            auth_manager,
            notification_manager,
            metrics_collector,
            event_bus,
            rate_limiter
        )
        
        logger.info("Application started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown application on shutdown."""
        await app_instance.stop()
        await app_instance.shutdown()
        logger.info("Application shutdown")
    
    return app, app_instance





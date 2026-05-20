"""
Startup and shutdown logic for the application.

Extracted from main.py for better organization and separation of concerns.
"""

import asyncio
import logging
from typing import Optional
from fastapi import FastAPI

from config.settings import settings, Environment
from core.plugin_system import get_plugin_registry, PluginType
from core.module_loader import get_module_loader
from core.service_factory import get_service_factory
from core.composition_root import get_composition_root
from api.services_locator import get_service_locator
from api.routers.router_manager import get_router_manager
from utils.cache import get_cache_manager
from utils.observability import setup_observability

logger = logging.getLogger(__name__)


async def initialize_application(app: FastAPI) -> None:
    """
    Initialize all application components.
    
    Args:
        app: FastAPI application instance to configure
        
    Raises:
        Exception: If initialization fails, all resources are cleaned up
    """
    logger.info("🚀 Starting Dermatology AI Service (v7.1.0 - Hexagonal Architecture)...")
    
    composition_root: Optional[object] = None
    
    try:
        composition_root = get_composition_root()
        config = _build_initialization_config()
        await composition_root.initialize(config)
        logger.info("✅ Composition root initialized")
        
        module_loader = get_module_loader()
        logger.info("✅ Module loader initialized")
        
        service_factory = get_service_factory()
        logger.info("✅ Service factory initialized")
        
        plugin_registry = get_plugin_registry()
        _load_plugins(plugin_registry)
        
        cache_manager = get_cache_manager()
        service_locator = get_service_locator()
        router_manager = get_router_manager()
        
        await cache_manager.initialize()
        logger.info("✅ Cache manager initialized")
        logger.info("✅ Service locator initialized (legacy)")
        
        if settings.environment == Environment.PRODUCTION:
            setup_observability()
            logger.info("✅ Observability initialized")
        
        await asyncio.gather(
            register_routers(app, router_manager, module_loader, composition_root),
            initialize_plugins(plugin_registry),
            return_exceptions=True
        )
        logger.info("✅ Routers registered")
        logger.info("✅ Plugins initialized")
        
        if not settings.debug:
            await warmup_services(service_factory, service_locator)
            logger.info("✅ Services warmed up")
        
        logger.info(f"✅ Service ready - Environment: {settings.environment.value}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        if composition_root:
            await composition_root.shutdown()
        raise


def _build_initialization_config() -> dict:
    """
    Build configuration dictionary for initialization.
    
    Returns:
        Configuration dictionary with database and message broker settings
    """
    database_type = (
        settings.database.type
        if hasattr(settings, "database") and hasattr(settings.database, "type")
        else "sqlite"
    )
    database_config = (
        settings.database.config
        if hasattr(settings, "database") and hasattr(settings.database, "config")
        else {}
    )
    
    return {
        "database_type": database_type,
        "database_config": database_config,
        "message_broker_type": "memory",
        "message_broker_config": {},
    }


def _load_plugins(plugin_registry: object) -> None:
    """
    Load plugins from directory.
    
    Args:
        plugin_registry: Plugin registry instance
    """
    try:
        plugin_registry.load_from_directory("plugins")
        logger.info("✅ Plugins discovered")
    except Exception as e:
        logger.warning(f"Plugin discovery skipped: {e}")


async def register_routers(
    app: FastAPI,
    router_manager: object,
    module_loader: object,
    composition_root: object
) -> None:
    """
    Register all application routers.
    
    Args:
        app: FastAPI application instance
        router_manager: Router manager instance
        module_loader: Module loader instance
        composition_root: Composition root instance
    """
    await _register_controllers(app, composition_root)
    _register_health_router(app)
    _register_modular_router(app, module_loader)
    _register_router_manager_routers(app, router_manager)


async def _register_controllers(app: FastAPI, composition_root: object) -> None:
    """
    Register controller routers.
    
    Args:
        app: FastAPI application instance
        composition_root: Composition root instance
    """
    try:
        from api.controllers.analysis_controller import AnalysisController
        from api.controllers.recommendation_controller import RecommendationController
        
        analyze_use_case = await composition_root.get_analyze_image_use_case()
        history_use_case = await composition_root.get_history_use_case()
        recommendations_use_case = await composition_root.get_recommendations_use_case()
        
        analysis_controller = AnalysisController(analyze_use_case, history_use_case)
        recommendation_controller = RecommendationController(recommendations_use_case)
        
        app.include_router(analysis_controller.router)
        app.include_router(recommendation_controller.router)
        logger.info("✅ Controllers registered")
    except Exception as e:
        logger.warning(f"Controller registration failed: {e}")


def _register_health_router(app: FastAPI) -> None:
    """
    Register health check router.
    
    Args:
        app: FastAPI application instance
    """
    try:
        from api.health import router as health_router
        app.include_router(health_router)
        logger.info("✅ Health check endpoints registered")
    except Exception as e:
        logger.warning(f"Health check registration failed: {e}")


def _register_modular_router(app: FastAPI, module_loader: object) -> None:
    """
    Register modular router.
    
    Args:
        app: FastAPI application instance
        module_loader: Module loader instance
    """
    try:
        modular_router_module = module_loader.load_module(
            "api.dermatology_api_modular",
            lazy=True
        )
        
        if modular_router_module:
            if hasattr(modular_router_module, "initialize_services"):
                modular_router_module.initialize_services()
            
            if hasattr(modular_router_module, "modular_router"):
                app.include_router(modular_router_module.modular_router)
                logger.info("✅ Modular router loaded")
    except Exception as e:
        logger.warning(f"Modular router loading failed, using fallback: {e}")
        _register_fallback_routers(app, module_loader)


def _register_fallback_routers(app: FastAPI, module_loader: object) -> None:
    """
    Register fallback routers.
    
    Args:
        app: FastAPI application instance
        module_loader: Module loader instance
    """
    try:
        legacy_router = module_loader.load_module("api.dermatology_api", lazy=False)
        if legacy_router and hasattr(legacy_router, "router"):
            app.include_router(legacy_router.router)
        
        admin_router = module_loader.load_module("services.admin_api", lazy=False)
        if admin_router and hasattr(admin_router, "router"):
            app.include_router(admin_router.router)
    except Exception as fallback_error:
        logger.error(f"Fallback router loading failed: {fallback_error}")


def _register_router_manager_routers(app: FastAPI, router_manager: object) -> None:
    """
    Register routers from router manager.
    
    Args:
        app: FastAPI application instance
        router_manager: Router manager instance
    """
    for router in router_manager.get_all_routers():
        app.include_router(router)


async def initialize_plugins(plugin_registry: object) -> None:
    """
    Initialize all registered plugins.
    
    Args:
        plugin_registry: Plugin registry instance
    """
    plugin_types = [
        PluginType.DATABASE,
        PluginType.CACHE,
        PluginType.MESSAGE_BROKER,
        PluginType.AUTHENTICATION,
        PluginType.OBSERVABILITY,
        PluginType.MIDDLEWARE,
        PluginType.ROUTER,
        PluginType.SERVICE,
    ]
    
    for plugin_type in plugin_types:
        plugins = plugin_registry.get_plugins_by_type(plugin_type)
        for plugin in plugins:
            plugin_id = f"{plugin.metadata.plugin_type.value}:{plugin.metadata.name}"
            await plugin_registry.initialize_plugin(plugin_id)


async def warmup_services(service_factory: object, service_locator: object) -> None:
    """
    Warm up critical services for faster first request.
    
    Args:
        service_factory: Service factory instance
        service_locator: Service locator instance
    """
    try:
        from utils.cache import get_cache_manager
        
        cache_manager = get_cache_manager()
        await cache_manager.ping()
        
        try:
            cache_service = await service_factory.create("cache")
            if hasattr(cache_service, "ping"):
                await cache_service.ping()
        except (ValueError, AttributeError):
            pass
        
        try:
            db_service = service_locator.get_service("database")
            if hasattr(db_service, "ping"):
                await db_service.ping()
        except (ValueError, AttributeError):
            pass
            
    except Exception as e:
        logger.debug(f"Warmup skipped: {e}")


async def shutdown_application() -> None:
    """
    Shutdown and cleanup all application components.
    
    Performs graceful shutdown of all services, plugins, and resources.
    """
    logger.info("🛑 Shutting down Dermatology AI Service...")
    
    try:
        composition_root = get_composition_root()
        await composition_root.shutdown()
        logger.info("✅ Composition root shutdown")
        
        plugin_registry = get_plugin_registry()
        await plugin_registry.shutdown_all()
        logger.info("✅ Plugins shutdown")
        
        cache_manager = get_cache_manager()
        await cache_manager.close()
        logger.info("✅ Cache manager closed")
        
        service_factory = get_service_factory()
        service_factory.clear_request_scope()
        logger.info("✅ Services cleaned up")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)
    
    logger.info("✅ Shutdown complete")


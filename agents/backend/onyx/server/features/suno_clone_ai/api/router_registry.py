"""
Router registry for centralized route management.

This module provides a clean way to register and organize all API routes
without manual includes scattered across files.
"""

import logging
from typing import List, Tuple, Optional
from fastapi import APIRouter

logger = logging.getLogger(__name__)


class RouterRegistry:
    """Centralized registry for all API routers."""
    
    def __init__(self):
        self._routers: List[Tuple[APIRouter, Optional[str]]] = []
        self._optional_routers: List[Tuple[str, Optional[str]]] = []
    
    def register(self, router: APIRouter, category: Optional[str] = None) -> None:
        """
        Register a router.
        
        Args:
            router: The APIRouter instance to register
            category: Optional category for organization
        """
        self._routers.append((router, category))
        logger.debug(f"Registered router: {router.prefix or 'root'} (category: {category})")
    
    def register_optional(self, module_path: str, category: Optional[str] = None) -> None:
        """
        Register an optional router that may not exist.
        
        Args:
            module_path: Import path to the router module (e.g., "api.routes.admin")
            category: Optional category for organization
        """
        self._optional_routers.append((module_path, category))
    
    def apply_to(self, main_router: APIRouter) -> None:
        """
        Apply all registered routers to the main router.
        
        Args:
            main_router: The main APIRouter to include all routers in
        """
        # Register required routers
        for router, category in self._routers:
            main_router.include_router(router)
            if category:
                logger.debug(f"Included router from category: {category}")
        
        # Register optional routers
        for module_path, category in self._optional_routers:
            try:
                module = __import__(module_path, fromlist=["router"])
                if hasattr(module, "router"):
                    main_router.include_router(module.router)
                    logger.debug(f"Included optional router: {module_path}")
                else:
                    logger.warning(f"Module {module_path} has no 'router' attribute")
            except ImportError as e:
                logger.debug(f"Optional router {module_path} not available: {e}")
            except Exception as e:
                logger.warning(f"Error loading optional router {module_path}: {e}")
        
        logger.info(f"Applied {len(self._routers)} required and {len(self._optional_routers)} optional routers")


# Global registry instance
_registry = RouterRegistry()


def get_registry() -> RouterRegistry:
    """Get the global router registry."""
    return _registry


def register_core_routes(registry: RouterRegistry) -> None:
    """Register core routes that are always available."""
    from .routes import (
        generation,
        songs,
        audio_processing,
        tags,
        comments,
        recommendations,
        favorites,
        export,
        search,
        playlists,
        sharing,
        stats,
        metrics,
        models,
        chat,
        health,
        performance
    )
    
    # Generation and processing
    registry.register(generation.router, "generation")
    registry.register(audio_processing.router, "generation")
    
    # Content management
    registry.register(songs.router, "content")
    registry.register(tags.router, "content")
    registry.register(comments.router, "content")
    registry.register(favorites.router, "content")
    registry.register(playlists.router, "content")
    
    # Discovery
    registry.register(search.router, "discovery")
    registry.register(recommendations.router, "discovery")
    
    # Social and sharing
    registry.register(sharing.router, "social")
    registry.register(stats.router, "social")
    
    # Administration
    registry.register(metrics.router, "admin")
    registry.register(models.router, "admin")
    registry.register(export.router, "admin")
    registry.register(chat.router, "admin")
    registry.register(health.router, "admin")
    registry.register(performance.router, "admin")


def register_optional_routes(registry: RouterRegistry) -> None:
    """Register optional routes that may not be available."""
    optional_modules = [
        "api.routes.admin",
        "api.routes.backup",
        "api.routes.analytics",
        "api.routes.webhooks",
        "api.routes.feature_flags",
        "api.routes.search_advanced",
        "api.routes.batch_processing",
        "api.routes.ab_testing",
        "api.routes.model_management",
        "api.routes.load_balancing",
        "api.routes.hyperparameter_tuning",
        "api.routes.transcription",
        "api.routes.sentiment",
        "api.routes.lyrics",
        "api.routes.distributed",
        "api.routes.scaling",
        "api.routes.streaming",
        "api.routes.audio_analysis",
        "api.routes.remix",
        "api.routes.karaoke",
        "api.routes.collaboration",
        "api.routes.marketplace",
        "api.routes.monetization",
        "api.routes.auto_dj",
        "api.routes.trends"
    ]
    
    for module_path in optional_modules:
        registry.register_optional(module_path, "optional")


"""
Composition Root - Dependency Injection Container.

Wires up all dependencies following Dependency Inversion Principle
and Clean Architecture patterns.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable

from .service_factory import get_service_factory, ServiceScope
from .domain.interfaces import (
    IAnalysisService,
    IRecommendationService,
)
from .infrastructure.adapters import IDatabaseAdapter
from .adapter_factory import AdapterFactory
from .repository_factory import RepositoryFactory
from .service_registration import ServiceRegistration
from .domain_service_factory import DomainServiceFactory
from .use_case_factory import UseCaseFactory
from .application import (
    AnalyzeImageUseCase,
    GetRecommendationsUseCase,
    GetAnalysisHistoryUseCase,
)

logger = logging.getLogger(__name__)


class CompositionRoot:
    """
    Composition root for dependency injection.
    
    Wires up all dependencies following Clean Architecture principles.
    Manages the lifecycle of all application dependencies and provides
    access to use cases with resolved dependencies.
    """
    
    def __init__(self) -> None:
        """
        Initialize composition root.
        
        Creates a new composition root instance with empty caches
        and uninitialized state.
        """
        self.service_factory = get_service_factory()
        self._initialized = False
        self._database_adapter: Optional[IDatabaseAdapter] = None
        self._config: Optional[Dict[str, Any]] = None
        self._use_case_cache: Dict[str, Any] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize composition root with configuration.
        
        Sets up all adapters, repositories, domain services, and use cases.
        This method should be called once during application startup.
        
        Args:
            config: Configuration dictionary with database and adapter settings
            
        Raises:
            Exception: If initialization fails, resources are cleaned up
        """
        if self._initialized:
            logger.warning("Composition root already initialized")
            return
        
        self._config = config
        
        try:
            database_adapter = await AdapterFactory.create_database_adapter(config)
            self._database_adapter = database_adapter
            
            adapters = await asyncio.gather(
                AdapterFactory.create_cache_adapter(config),
                AdapterFactory.create_image_processor_adapter(config),
                AdapterFactory.create_event_publisher_adapter(config)
            )
            
            cache_adapter, image_processor_adapter, event_publisher_adapter = adapters
            
            repositories = RepositoryFactory.create_repositories(database_adapter)
            ServiceRegistration.register_repositories(self.service_factory, repositories)
            
            ServiceRegistration.register_adapters(
                self.service_factory,
                {
                    "image_processor": image_processor_adapter,
                    "cache": cache_adapter,
                    "event_publisher": event_publisher_adapter,
                }
            )
            
            domain_services = await asyncio.gather(
                DomainServiceFactory.create_analysis_service(self.service_factory),
                DomainServiceFactory.create_recommendation_service(self.service_factory),
                return_exceptions=True
            )
            
            analysis_service, recommendation_service = domain_services
            
            ServiceRegistration.register_domain_services(
                self.service_factory,
                {
                    "analysis_service": analysis_service,
                    "recommendation_service": recommendation_service,
                }
            )
            
            self._initialized = True
            logger.info("✅ Composition root initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize composition root: {e}", exc_info=True)
            await self._cleanup()
            raise
    
    async def _cleanup(self) -> None:
        """
        Cleanup resources on initialization failure.
        
        Safely closes database adapter and other resources
        if initialization fails partway through.
        """
        if self._database_adapter:
            try:
                if hasattr(self._database_adapter, "close"):
                    await self._database_adapter.close()
            except Exception as e:
                logger.error(f"Error closing database adapter: {e}")
    
    
    async def _get_cached_use_case(
        self,
        cache_key: str,
        factory_method: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """
        Get use case from cache or create it using factory method.
        
        Args:
            cache_key: Key to identify the use case in cache
            factory_method: Async factory function to create the use case
            
        Returns:
            Use case instance (cached or newly created)
            
        Raises:
            RuntimeError: If composition root is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Composition root not initialized")
        
        if cache_key in self._use_case_cache:
            return self._use_case_cache[cache_key]
        
        use_case = await factory_method(self.service_factory)
        self._use_case_cache[cache_key] = use_case
        return use_case
    
    async def get_analyze_image_use_case(self) -> AnalyzeImageUseCase:
        """
        Get analyze image use case with dependencies resolved.
        
        Returns:
            Configured AnalyzeImageUseCase instance
            
        Raises:
            RuntimeError: If composition root is not initialized
        """
        return await self._get_cached_use_case(
            "analyze_image_use_case",
            UseCaseFactory.create_analyze_image_use_case
        )
    
    async def get_recommendations_use_case(self) -> GetRecommendationsUseCase:
        """
        Get recommendations use case with dependencies resolved.
        
        Returns:
            Configured GetRecommendationsUseCase instance
            
        Raises:
            RuntimeError: If composition root is not initialized
        """
        return await self._get_cached_use_case(
            "recommendations_use_case",
            UseCaseFactory.create_recommendations_use_case
        )
    
    async def get_history_use_case(self) -> GetAnalysisHistoryUseCase:
        """
        Get history use case with dependencies resolved.
        
        Returns:
            Configured GetAnalysisHistoryUseCase instance
            
        Raises:
            RuntimeError: If composition root is not initialized
        """
        return await self._get_cached_use_case(
            "history_use_case",
            UseCaseFactory.create_history_use_case
        )
    
    async def shutdown(self) -> None:
        """
        Shutdown and cleanup resources.
        
        Closes all adapters, clears caches, and resets initialization state.
        Should be called during application shutdown.
        """
        if self._database_adapter:
            try:
                if hasattr(self._database_adapter, "close"):
                    await self._database_adapter.close()
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        
        self.service_factory.clear_request_scope()
        self._use_case_cache.clear()
        self._initialized = False
        logger.info("✅ Composition root shutdown")


# Global composition root singleton
_composition_root: Optional[CompositionRoot] = None


def get_composition_root() -> CompositionRoot:
    """
    Get or create global composition root singleton.
    
    Returns:
        Global CompositionRoot instance (singleton pattern)
    """
    global _composition_root
    if _composition_root is None:
        _composition_root = CompositionRoot()
    return _composition_root


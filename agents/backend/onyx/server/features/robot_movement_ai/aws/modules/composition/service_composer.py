"""
Service Composer
================

Composes services from modular components using dependency injection.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI
from aws.modules.data.repository_factory import RepositoryFactory
from aws.modules.data.cache_factory import CacheFactory
from aws.modules.data.messaging_factory import MessagingFactory
from aws.modules.business.service_factory import ServiceFactory
from aws.modules.presentation.api_router import APIRouter
from aws.modules.ports.repository_port import RepositoryPort
from aws.modules.ports.cache_port import CachePort
from aws.modules.ports.messaging_port import MessagingPort

logger = logging.getLogger(__name__)


class ServiceComposer:
    """Composes services from modular components."""
    
    def __init__(
        self,
        service_name: str,
        repository: Optional[RepositoryPort] = None,
        cache: Optional[CachePort] = None,
        messaging: Optional[MessagingPort] = None
    ):
        self.service_name = service_name
        self.repository = repository
        self.cache = cache
        self.messaging = messaging
        self._app: Optional[FastAPI] = None
    
    @classmethod
    def from_config(cls, service_name: str, config: Dict[str, Any]) -> "ServiceComposer":
        """Create composer from configuration."""
        # Create adapters from config
        repo_type = config.get("repository", {}).get("type", "dynamodb")
        repository = RepositoryFactory.create(
            adapter_type=repo_type,
            table_name=config.get("repository", {}).get("table_name")
        )
        
        cache_type = config.get("cache", {}).get("type", "redis")
        cache = CacheFactory.create(
            adapter_type=cache_type,
            redis_url=config.get("cache", {}).get("redis_url")
        )
        
        messaging_type = config.get("messaging", {}).get("type")
        messaging = None
        if messaging_type:
            messaging = MessagingFactory.create(
                adapter_type=messaging_type,
                kafka_servers=config.get("messaging", {}).get("kafka_servers")
            )
        
        return cls(
            service_name=service_name,
            repository=repository,
            cache=cache,
            messaging=messaging
        )
    
    @classmethod
    def from_env(cls, service_name: str) -> "ServiceComposer":
        """Create composer from environment variables."""
        repository = RepositoryFactory.create_from_env(service_name)
        cache = CacheFactory.create_from_env()
        messaging = MessagingFactory.create_from_env()
        
        return cls(
            service_name=service_name,
            repository=repository,
            cache=cache,
            messaging=messaging
        )
    
    def create_business_layer(self) -> ServiceFactory:
        """Create business layer with dependencies."""
        return ServiceFactory(
            repository=self.repository,
            cache=self.cache,
            messaging=self.messaging
        )
    
    def create_presentation_layer(self, prefix: str = "/api/v1") -> APIRouter:
        """Create presentation layer."""
        return APIRouter(prefix=prefix, tags=[self.service_name])
    
    def compose_service(self, routers: list) -> FastAPI:
        """Compose complete service from components."""
        app = FastAPI(
            title=f"{self.service_name.title()} Service",
            description=f"Modular {self.service_name} microservice",
            version="1.0.0"
        )
        
        # Include routers
        for router in routers:
            app.include_router(router.get_router())
        
        # Store dependencies in app state
        app.state.repository = self.repository
        app.state.cache = self.cache
        app.state.messaging = self.messaging
        app.state.service_factory = self.create_business_layer()
        
        self._app = app
        logger.info(f"Composed service: {self.service_name}")
        
        return app
    
    def get_app(self) -> Optional[FastAPI]:
        """Get composed FastAPI app."""
        return self._app
















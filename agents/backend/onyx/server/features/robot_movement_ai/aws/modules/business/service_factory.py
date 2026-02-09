"""
Service Factory
===============

Factory for creating domain services with dependency injection.
"""

import logging
from typing import Dict, Any, Optional
from aws.modules.business.domain_services import (
    DomainService,
    MovementDomainService,
    TrajectoryDomainService
)
from aws.modules.ports.repository_port import RepositoryPort
from aws.modules.ports.cache_port import CachePort
from aws.modules.ports.messaging_port import MessagingPort

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory for creating domain services."""
    
    def __init__(
        self,
        repository: Optional[RepositoryPort] = None,
        cache: Optional[CachePort] = None,
        messaging: Optional[MessagingPort] = None
    ):
        self.repository = repository
        self.cache = cache
        self.messaging = messaging
        self._services: Dict[str, DomainService] = {}
    
    def create_movement_service(self) -> MovementDomainService:
        """Create movement domain service."""
        if "movement" not in self._services:
            self._services["movement"] = MovementDomainService(
                repository=self.repository,
                cache=self.cache,
                messaging=self.messaging
            )
        return self._services["movement"]
    
    def create_trajectory_service(self) -> TrajectoryDomainService:
        """Create trajectory domain service."""
        if "trajectory" not in self._services:
            self._services["trajectory"] = TrajectoryDomainService(
                repository=self.repository,
                cache=self.cache,
                messaging=self.messaging
            )
        return self._services["trajectory"]
    
    def get_service(self, name: str) -> Optional[DomainService]:
        """Get service by name."""
        return self._services.get(name)
    
    def register_service(self, name: str, service: DomainService):
        """Register a custom service."""
        self._services[name] = service
        logger.info(f"Registered domain service: {name}")
















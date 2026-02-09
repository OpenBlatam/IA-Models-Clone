"""
Infrastructure Factory - Factory para infraestructura
=====================================================

Factory que crea instancias de servicios de infraestructura.
"""

import logging
from typing import Optional

from ..interfaces.cache import ICacheService
from ..interfaces.events import IEventPublisher, IEventSubscriber
from ..interfaces.workers import IWorkerService
from ..infrastructure.cache import CacheService
from ..infrastructure.events import EventPublisher, EventSubscriber
from ..infrastructure.workers import WorkerService

logger = logging.getLogger(__name__)


class InfrastructureFactory:
    """Factory para crear servicios de infraestructura"""
    
    @staticmethod
    def create_cache_service(
        backend: Optional[str] = None,
        url: Optional[str] = None
    ) -> ICacheService:
        """
        Crea servicio de cache.
        
        Args:
            backend: Backend de cache (redis, in_memory)
            url: URL del cache (para Redis)
        
        Returns:
            Servicio de cache
        """
        return CacheService()
    
    @staticmethod
    def create_event_publisher() -> IEventPublisher:
        """
        Crea publicador de eventos.
        
        Returns:
            Publicador de eventos
        """
        return EventPublisher()
    
    @staticmethod
    def create_event_subscriber() -> IEventSubscriber:
        """
        Crea suscriptor de eventos.
        
        Returns:
            Suscriptor de eventos
        """
        return EventSubscriber()
    
    @staticmethod
    def create_worker_service() -> IWorkerService:
        """
        Crea servicio de workers.
        
        Returns:
            Servicio de workers
        """
        return WorkerService()
















"""
Base Service - Servicio base para todos los servicios
=====================================================

Clase base que proporciona funcionalidad común a todos los servicios.
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC

from ..interfaces.cache import ICacheService
from ..interfaces.events import IEventPublisher
from ..debug.debug_logger import get_debug_logger

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Clase base para todos los servicios.
    
    Proporciona:
    - Cache service común
    - Event publisher común
    - Debug logger común
    - Métodos helper comunes
    """
    
    def __init__(
        self,
        cache_service: Optional[ICacheService] = None,
        event_publisher: Optional[IEventPublisher] = None,
        service_name: Optional[str] = None
    ):
        """
        Args:
            cache_service: Servicio de cache
            event_publisher: Publicador de eventos
            service_name: Nombre del servicio
        """
        self.cache_service = cache_service
        self.event_publisher = event_publisher
        self.service_name = service_name or self.__class__.__name__
        self.debug_logger = get_debug_logger()
    
    async def _publish_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """
        Publica evento si event_publisher está disponible.
        
        Args:
            event_type: Tipo de evento
            event_data: Datos del evento
        
        Returns:
            True si se publicó exitosamente
        """
        if self.event_publisher:
            return await self.event_publisher.publish(event_type, event_data)
        return False
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Obtiene valor del cache.
        
        Args:
            key: Clave del cache
        
        Returns:
            Valor del cache o None
        """
        if self.cache_service:
            return await self.cache_service.get(key)
        return None
    
    async def _set_to_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Establece valor en cache.
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: Tiempo de vida (segundos)
        
        Returns:
            True si se guardó exitosamente
        """
        if self.cache_service:
            return await self.cache_service.set(key, value, ttl)
        return False
    
    async def _delete_from_cache(self, key: str) -> bool:
        """
        Elimina valor del cache.
        
        Args:
            key: Clave del cache
        
        Returns:
            True si se eliminó exitosamente
        """
        if self.cache_service:
            return await self.cache_service.delete(key)
        return False
    
    def _log_service_call(
        self,
        method_name: str,
        duration: float,
        success: bool,
        **kwargs
    ):
        """
        Log de llamada a servicio.
        
        Args:
            method_name: Nombre del método
            duration: Duración en segundos
            success: Si fue exitoso
            **kwargs: Datos adicionales
        """
        self.debug_logger.log_service_call(
            service_name=self.service_name,
            method_name=method_name,
            duration=duration,
            success=success,
            **kwargs
        )
















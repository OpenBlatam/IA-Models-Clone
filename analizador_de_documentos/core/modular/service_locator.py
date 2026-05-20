"""
Service Locator - Localizador de Servicios
==========================================

Patrón Service Locator para inyección de dependencias.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServiceInfo:
    """Información de servicio."""
    service_id: str
    service_type: Type
    instance: Any
    singleton: bool = True
    factory: Optional[Callable] = None


class ServiceLocator:
    """Localizador de servicios."""
    
    def __init__(self):
        """Inicializar localizador."""
        self.services: Dict[str, ServiceInfo] = {}
        self.singletons: Dict[str, Any] = {}
    
    def register(
        self,
        service_id: str,
        service_type: Type[T],
        instance: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None,
        singleton: bool = True
    ):
        """Registrar servicio."""
        if service_id in self.services:
            logger.warning(f"Servicio {service_id} ya registrado, actualizando...")
        
        self.services[service_id] = ServiceInfo(
            service_id=service_id,
            service_type=service_type,
            instance=instance,
            singleton=singleton,
            factory=factory
        )
        
        if singleton and instance:
            self.singletons[service_id] = instance
        
        logger.info(f"Servicio registrado: {service_id}")
    
    def get(self, service_id: str) -> Optional[Any]:
        """Obtener servicio."""
        service_info = self.services.get(service_id)
        if not service_info:
            return None
        
        # Si es singleton y ya existe instancia, retornarla
        if service_info.singleton and service_id in self.singletons:
            return self.singletons[service_id]
        
        # Si hay instancia directa
        if service_info.instance:
            if service_info.singleton:
                self.singletons[service_id] = service_info.instance
            return service_info.instance
        
        # Si hay factory, crear instancia
        if service_info.factory:
            instance = service_info.factory()
            if service_info.singleton:
                self.singletons[service_id] = instance
            return instance
        
        return None
    
    def get_by_type(self, service_type: Type[T]) -> Optional[T]:
        """Obtener servicio por tipo."""
        for service_info in self.services.values():
            if service_info.service_type == service_type:
                return self.get(service_info.service_id)
        return None
    
    def unregister(self, service_id: str):
        """Desregistrar servicio."""
        if service_id in self.services:
            del self.services[service_id]
            if service_id in self.singletons:
                del self.singletons[service_id]
            logger.info(f"Servicio desregistrado: {service_id}")
    
    def has_service(self, service_id: str) -> bool:
        """Verificar si servicio está registrado."""
        return service_id in self.services
    
    def list_services(self) -> List[str]:
        """Listar IDs de servicios."""
        return list(self.services.keys())


__all__ = [
    "ServiceLocator",
    "ServiceInfo"
]



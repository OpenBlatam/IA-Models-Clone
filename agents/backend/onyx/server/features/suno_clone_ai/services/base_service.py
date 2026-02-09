"""
Clase Base para Servicios

Proporciona funcionalidad común para todos los servicios.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC
from datetime import datetime
from core.events import get_event_bus, Event, EventType


class BaseService(ABC):
    """Clase base para todos los servicios"""
    
    def __init__(self, service_name: str):
        """
        Args:
            service_name: Nombre del servicio
        """
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
        self.event_bus = get_event_bus()
        self.initialized = False
        self.initialized_at: Optional[datetime] = None
        self.logger.info(f"{service_name} service created")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Inicializa el servicio
        
        Args:
            config: Configuración opcional
        
        Returns:
            True si se inicializó exitosamente
        """
        if self.initialized:
            return True
        
        try:
            success = await self._on_initialize(config or {})
            if success:
                self.initialized = True
                self.initialized_at = datetime.now()
                self.logger.info(f"{self.service_name} initialized")
            return success
        except Exception as e:
            self.logger.error(f"Error initializing {self.service_name}: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> None:
        """Cierra el servicio"""
        if not self.initialized:
            return
        
        try:
            await self._on_shutdown()
            self.initialized = False
            self.logger.info(f"{self.service_name} shut down")
        except Exception as e:
            self.logger.error(f"Error shutting down {self.service_name}: {e}", exc_info=True)
    
    async def _on_initialize(self, config: Dict[str, Any]) -> bool:
        """
        Hook para inicialización personalizada
        
        Args:
            config: Configuración
        
        Returns:
            True si se inicializó exitosamente
        """
        return True
    
    async def _on_shutdown(self) -> None:
        """Hook para cierre personalizado"""
        pass
    
    def _publish_event(self, event_type: EventType, data: Dict[str, Any]):
        """
        Publica un evento
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=self.service_name
        )
        # Publicar de forma asíncrona sin esperar
        import asyncio
        asyncio.create_task(self.event_bus.publish(event))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del servicio
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "service_name": self.service_name,
            "initialized": self.initialized,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None
        }


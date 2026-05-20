"""
Event Service - Servicio de Eventos
====================================

Sistema de eventos para desacoplamiento.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    POST_CREATED = "post_created"
    POST_PUBLISHED = "post_published"
    POST_FAILED = "post_failed"
    MEME_ADDED = "meme_added"
    TEMPLATE_CREATED = "template_created"
    PLATFORM_CONNECTED = "platform_connected"
    ANALYTICS_UPDATED = "analytics_updated"
    BACKUP_CREATED = "backup_created"


class EventService:
    """Servicio de eventos"""
    
    def __init__(self):
        """Inicializar servicio de eventos"""
        self.listeners: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict[str, Any]] = []
        logger.info("Event Service inicializado")
    
    def subscribe(self, event_type: EventType, handler: Callable[[Dict[str, Any]], None]):
        """
        Suscribirse a un evento
        
        Args:
            event_type: Tipo de evento
            handler: Función que maneja el evento
        """
        self.listeners[event_type].append(handler)
        logger.info(f"Handler suscrito a {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """
        Desuscribirse de un evento
        
        Args:
            event_type: Tipo de evento
            handler: Handler a remover
        """
        if handler in self.listeners[event_type]:
            self.listeners[event_type].remove(handler)
            logger.info(f"Handler desuscrito de {event_type.value}")
    
    def emit(self, event_type: EventType, data: Dict[str, Any]):
        """
        Emitir un evento
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        event = {
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_history.append(event)
        
        # Ejecutar handlers
        handlers = self.listeners.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error en handler de evento {event_type.value}: {e}")
        
        logger.debug(f"Evento emitido: {event_type.value}")
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de eventos
        
        Args:
            event_type: Filtrar por tipo
            limit: Límite de eventos
            
        Returns:
            Lista de eventos
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.get("type") == event_type.value]
        
        return events[-limit:]
    
    def clear_history(self):
        """Limpiar historial de eventos"""
        self.event_history.clear()
        logger.info("Historial de eventos limpiado")




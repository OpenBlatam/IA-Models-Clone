"""
Sistema de eventos para notificaciones y broadcasting.
"""

from typing import Dict, Any, List, Callable, Optional
from enum import Enum
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Tipos de eventos."""
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    REPOSITORY_ACCESSED = "repository_accessed"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Event:
    """Representa un evento."""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """
        Inicializar evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento (opcional)
            timestamp: Timestamp del evento (opcional)
        """
        self.event_type = event_type
        self.data = data
        self.source = source or "system"
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir evento a diccionario."""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp
        }


class EventBus:
    """Bus de eventos para pub/sub."""
    
    def __init__(self):
        """Inicializar bus de eventos."""
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000  # Mantener últimos 1000 eventos
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        Suscribirse a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función que maneja el evento
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        Desuscribirse de un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función a remover
        """
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from {event_type.value}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type.value}")
    
    async def publish(self, event: Event) -> None:
        """
        Publicar un evento.
        
        Args:
            event: Evento a publicar
        """
        # Agregar a historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Broadcast WebSocket automático para eventos relevantes
        try:
            # Importación lazy para evitar imports circulares
            import sys
            if 'api.routes.websocket_routes' in sys.modules:
                from api.routes.websocket_routes import manager
                
                if event.event_type in [
                    EventType.TASK_CREATED,
                    EventType.TASK_UPDATED,
                    EventType.TASK_COMPLETED,
                    EventType.TASK_FAILED
                ]:
                    # Broadcast a clientes suscritos a tareas
                    await manager.broadcast(event.to_dict(), "tasks")
                
                elif event.event_type in [
                    EventType.AGENT_STARTED,
                    EventType.AGENT_STOPPED,
                    EventType.AGENT_PAUSED,
                    EventType.AGENT_RESUMED,
                    EventType.AGENT_STATUS_CHANGED
                ]:
                    # Broadcast a clientes suscritos al agente
                    await manager.broadcast(event.to_dict(), "agent")
                
                # Broadcast general
                await manager.broadcast_to_all(event.to_dict())
        except (ImportError, AttributeError) as e:
            logger.debug(f"WebSocket broadcast failed (non-critical): {e}")
        except Exception as e:
            logger.debug(f"WebSocket broadcast error (non-critical): {e}")
        
        # Notificar suscriptores
        handlers = self.subscribers.get(event.event_type, [])
        all_handlers = self.subscribers.get(EventType.INFO, [])  # Handlers globales
        
        for handler in handlers + all_handlers:
            try:
                if callable(handler):
                    import asyncio
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type.value}: {e}", exc_info=True)
        
        logger.debug(f"Published event: {event.event_type.value}")
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Obtener historial de eventos.
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
    
    def clear_history(self) -> None:
        """Limpiar historial de eventos."""
        self.event_history.clear()
        logger.info("Event history cleared")


# Instancia global del bus de eventos
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Obtener instancia global del bus de eventos."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Funciones helper para publicar eventos comunes
async def publish_task_event(
    event_type: EventType,
    task: Dict[str, Any],
    source: Optional[str] = None
) -> None:
    """Publicar evento relacionado con tarea."""
    event = Event(
        event_type=event_type,
        data={"task": task},
        source=source or "task_processor"
    )
    await get_event_bus().publish(event)


async def publish_agent_event(
    event_type: EventType,
    agent_state: Dict[str, Any],
    source: Optional[str] = None
) -> None:
    """Publicar evento relacionado con agente."""
    event = Event(
        event_type=event_type,
        data={"agent": agent_state},
        source=source or "agent_routes"
    )
    await get_event_bus().publish(event)


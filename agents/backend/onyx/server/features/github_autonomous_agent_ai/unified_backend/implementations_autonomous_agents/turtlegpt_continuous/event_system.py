"""
Event System Module
==================

Sistema de eventos para desacoplar componentes del agente.
Permite comunicación asíncrona entre componentes sin acoplamiento directo.
"""

import asyncio
import logging
from typing import Dict, Any, Callable, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos del sistema."""
    TASK_SUBMITTED = "task_submitted"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    REFLECTION_TRIGGERED = "reflection_triggered"
    LEARNING_OPPORTUNITY = "learning_opportunity"
    STRATEGY_SELECTED = "strategy_selected"
    MEMORY_UPDATED = "memory_updated"
    METRICS_UPDATED = "metrics_updated"
    ERROR_OCCURRED = "error_occurred"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"


@dataclass
class Event:
    """Representa un evento del sistema."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir evento a diccionario."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


class EventBus:
    """
    Bus de eventos para comunicación desacoplada entre componentes.
    
    Implementa el patrón Observer/Pub-Sub para permitir que los componentes
    se comuniquen sin conocer directamente a otros componentes.
    """
    
    def __init__(self):
        """Inicializar bus de eventos."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history_size = 1000
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ):
        """
        Suscribirse a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            callback: Función a llamar cuando ocurra el evento
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type.value}")
    
    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ):
        """
        Desuscribirse de un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            callback: Función a remover
        """
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
    
    async def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None
    ):
        """
        Publicar un evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento (opcional)
        """
        event = Event(
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        # Guardar en historial
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
        
        # Notificar suscriptores
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(
                        f"Error in event callback for {event_type.value}: {e}",
                        exc_info=True
                    )
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Obtener historial de eventos.
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de eventos a retornar
            
        Returns:
            Lista de eventos
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del bus de eventos."""
        stats = {
            "total_events": len(self._event_history),
            "subscribers": {
                event_type.value: len(callbacks)
                for event_type, callbacks in self._subscribers.items()
            },
            "event_counts": {}
        }
        
        # Contar eventos por tipo
        for event in self._event_history:
            event_type_str = event.event_type.value
            stats["event_counts"][event_type_str] = (
                stats["event_counts"].get(event_type_str, 0) + 1
            )
        
        return stats

"""
Event Bus - Sistema de eventos
================================

Sistema de eventos pub/sub para comunicación entre componentes.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    TASK_ADDED = "task_added"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"
    COMMAND_RECEIVED = "command_received"
    SCHEDULED_TASK_EXECUTED = "scheduled_task_executed"
    BACKUP_CREATED = "backup_created"
    PLUGIN_LOADED = "plugin_loaded"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """Evento"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None


class EventBus:
    """
    Bus de eventos pub/sub con mejoras de rendimiento y filtrado.
    
    Características:
    - Suscripciones por tipo de evento
    - Historial de eventos con límite configurable
    - Filtrado de eventos por tipo y tiempo
    - Manejo robusto de errores en callbacks
    - Estadísticas de eventos
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Inicializar event bus.
        
        Args:
            max_history: Número máximo de eventos a mantener en historial
        """
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = max_history
        self._event_counters: Dict[EventType, int] = {}
        self._error_count = 0
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Suscribirse a un tipo de evento"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logger.debug(f"📡 Subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Desuscribirse de un tipo de evento"""
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                logger.debug(f"📡 Unsubscribed from {event_type.value}")
    
    async def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> None:
        """
        Publicar evento a todos los suscriptores.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento (opcional)
        """
        event = Event(
            type=event_type,
            data=data,
            source=source
        )
        
        # Actualizar contador
        self._event_counters[event_type] = self._event_counters.get(event_type, 0) + 1
        
        # Agregar al historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notificar suscriptores con manejo robusto de errores
        if event_type in self.subscribers:
            callbacks = list(self.subscribers[event_type])  # Copia para evitar modificaciones durante iteración
            
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await asyncio.wait_for(callback(event), timeout=30.0)
                    else:
                        callback(event)
                except asyncio.TimeoutError:
                    logger.warning(f"Event callback timeout for {event_type.value}")
                    self._error_count += 1
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type.value}: {e}", exc_info=True)
                    self._error_count += 1
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Event]:
        """
        Obtener eventos del historial con filtrado.
        
        Args:
            event_type: Filtrar por tipo de evento (opcional)
            limit: Número máximo de eventos a retornar
            since: Filtrar eventos desde esta fecha (opcional)
            
        Returns:
            Lista de eventos filtrados y ordenados
        """
        events = list(self.event_history)
        
        # Filtrar por tipo
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        # Filtrar por fecha
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Ordenar por timestamp (más recientes primero)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del event bus.
        
        Returns:
            Diccionario con estadísticas detalladas
        """
        by_type = {}
        for event in self.event_history:
            event_type = event.type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1
        
        total_subscribers = sum(len(callbacks) for callbacks in self.subscribers.values())
        
        return {
            "total_events": len(self.event_history),
            "total_published": sum(self._event_counters.values()),
            "subscribers": total_subscribers,
            "subscribers_by_type": {
                event_type.value: len(callbacks)
                for event_type, callbacks in self.subscribers.items()
            },
            "events_by_type": by_type,
            "error_count": self._error_count,
            "max_history": self.max_history,
            "history_usage_percent": round((len(self.event_history) / self.max_history) * 100, 2) if self.max_history > 0 else 0
        }
    
    def clear_history(self, event_type: Optional[EventType] = None) -> int:
        """
        Limpiar historial de eventos.
        
        Args:
            event_type: Si se especifica, solo limpiar eventos de este tipo
            
        Returns:
            Número de eventos eliminados
        """
        if event_type:
            before = len(self.event_history)
            self.event_history = [e for e in self.event_history if e.type != event_type]
            return before - len(self.event_history)
        else:
            count = len(self.event_history)
            self.event_history.clear()
            return count
    
    def get_recent_events(
        self,
        event_type: Optional[EventType] = None,
        minutes: int = 5
    ) -> List[Event]:
        """
        Obtener eventos recientes de los últimos N minutos.
        
        Args:
            event_type: Filtrar por tipo de evento (opcional)
            minutes: Número de minutos hacia atrás
            
        Returns:
            Lista de eventos recientes
        """
        from datetime import timedelta
        since = datetime.now() - timedelta(minutes=minutes)
        return self.get_events(event_type=event_type, since=since, limit=1000)



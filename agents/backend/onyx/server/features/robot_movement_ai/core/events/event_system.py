"""
Event System
============

Sistema de eventos para comunicación entre componentes.
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos."""
    TRAJECTORY_OPTIMIZED = "trajectory_optimized"
    MOVEMENT_STARTED = "movement_started"
    MOVEMENT_COMPLETED = "movement_completed"
    MOVEMENT_FAILED = "movement_failed"
    OBSTACLE_DETECTED = "obstacle_detected"
    ERROR_OCCURRED = "error_occurred"
    CONFIG_CHANGED = "config_changed"
    METRIC_UPDATED = "metric_updated"


@dataclass
class Event:
    """Evento del sistema."""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class EventEmitter:
    """
    Emisor de eventos.
    
    Permite emitir eventos y registrar listeners.
    """
    
    def __init__(self):
        """Inicializar emisor de eventos."""
        self.listeners: Dict[EventType, List[Callable]] = {}
        self.once_listeners: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def on(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Registrar listener permanente.
        
        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
        logger.debug(f"Registered listener for {event_type.value}")
    
    def once(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Registrar listener de una sola vez.
        
        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        if event_type not in self.once_listeners:
            self.once_listeners[event_type] = []
        self.once_listeners[event_type].append(callback)
    
    def off(self, event_type: EventType, callback: Optional[Callable] = None) -> None:
        """
        Remover listener.
        
        Args:
            event_type: Tipo de evento
            callback: Callback a remover (None = remover todos)
        """
        if callback is None:
            self.listeners.pop(event_type, None)
        elif event_type in self.listeners:
            if callback in self.listeners[event_type]:
                self.listeners[event_type].remove(callback)
    
    def emit(self, event_type: EventType, data: Optional[Dict[str, Any]] = None, source: str = "") -> None:
        """
        Emitir evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source
        )
        
        # Agregar a historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notificar listeners permanentes
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event listener for {event_type.value}: {e}")
        
        # Notificar listeners de una vez
        if event_type in self.once_listeners:
            callbacks = self.once_listeners.pop(event_type)
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in once listener for {event_type.value}: {e}")
    
    async def emit_async(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: str = ""
    ) -> None:
        """
        Emitir evento de forma asíncrona.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source
        )
        
        # Agregar a historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notificar listeners permanentes
        if event_type in self.listeners:
            tasks = []
            for callback in self.listeners[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(event))
                else:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in async event listener: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Notificar listeners de una vez
        if event_type in self.once_listeners:
            callbacks = self.once_listeners.pop(event_type)
            tasks = []
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(event))
                else:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in async once listener: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_event_history(
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
        events = self.event_history[-limit:] if limit else self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events


# Instancia global
_event_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Obtener instancia global del emisor de eventos."""
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = EventEmitter()
    return _event_emitter







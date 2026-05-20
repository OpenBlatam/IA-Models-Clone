"""
MCP Events - Sistema de eventos para el servidor MCP
====================================================

Sistema pub/sub para eventos internos del servidor MCP.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    COMMAND_EXECUTED = "command_executed"
    COMMAND_FAILED = "command_failed"
    COMMAND_CACHED = "command_cached"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WEBSOCKET_CONNECTED = "websocket_connected"
    WEBSOCKET_DISCONNECTED = "websocket_disconnected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"


@dataclass
class Event:
    """Evento del sistema"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None


class EventBus:
    """Sistema de eventos pub/sub"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_type: EventType, handler: Callable):
        """Suscribirse a un tipo de evento"""
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed to {event_type.value}")
    
    async def unsubscribe(self, event_type: EventType, handler: Callable):
        """Desuscribirse de un tipo de evento"""
        async with self._lock:
            if event_type in self._subscribers:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)
                    logger.debug(f"Unsubscribed from {event_type.value}")
    
    async def publish(self, event: Event):
        """Publicar un evento"""
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        
        handlers = self._subscribers.get(event.event_type, [])
        
        if handlers:
            tasks = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type.value}: {e}", exc_info=True)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Published event: {event.event_type.value}")
    
    async def get_recent_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Obtener eventos recientes"""
        async with self._lock:
            events = self._event_history.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    async def clear_history(self):
        """Limpiar historial de eventos"""
        async with self._lock:
            self._event_history.clear()
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Obtener número de suscriptores para un tipo de evento"""
        return len(self._subscribers.get(event_type, []))


class EventHandler:
    """Manejador base para eventos"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def handle(self, event: Event):
        """Manejar evento (debe ser implementado por subclases)"""
        raise NotImplementedError


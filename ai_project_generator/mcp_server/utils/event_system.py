"""
Event System - Sistema de eventos/pub-sub avanzado
===================================================

Sistema de eventos con pub-sub, filtros, y manejo asíncrono.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from collections import defaultdict
from threading import Lock
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class EventPriority(str, Enum):
    """Prioridad de evento."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Evento."""
    
    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "metadata": self.metadata,
        }


class EventHandler:
    """Handler de eventos."""
    
    def __init__(
        self,
        handler: Union[Callable[[Event], Any], Callable[[Event], Awaitable[Any]]],
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: int = 0
    ):
        """
        Inicializar handler.
        
        Args:
            handler: Función handler (síncrona o asíncrona)
            filter_func: Función de filtro (opcional)
            priority: Prioridad del handler (mayor = más prioritario)
        """
        self.handler = handler
        self.filter_func = filter_func
        self.priority = priority
        self._is_async = asyncio.iscoroutinefunction(handler)
    
    def should_handle(self, event: Event) -> bool:
        """
        Verificar si el handler debe procesar el evento.
        
        Args:
            event: Evento a verificar
        
        Returns:
            True si debe procesar, False en caso contrario
        """
        if self.filter_func:
            return self.filter_func(event)
        return True
    
    async def handle(self, event: Event) -> Any:
        """
        Procesar evento.
        
        Args:
            event: Evento a procesar
        
        Returns:
            Resultado del handler
        """
        if self._is_async:
            return await self.handler(event)
        else:
            return self.handler(event)


class EventBus:
    """
    Bus de eventos con pub-sub.
    
    Permite publicar eventos y suscribirse a ellos con filtros.
    """
    
    def __init__(self):
        """Inicializar bus de eventos."""
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._lock = Lock()
        self._event_history: List[Event] = []
        self._max_history: int = 1000
    
    def subscribe(
        self,
        event_type: Optional[str] = None,
        handler: Optional[Union[Callable[[Event], Any], Callable[[Event], Awaitable[Any]]]] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: int = 0
    ) -> EventHandler:
        """
        Suscribirse a eventos.
        
        Args:
            event_type: Tipo de evento (None = todos)
            handler: Función handler
            filter_func: Función de filtro (opcional)
            priority: Prioridad del handler
        
        Returns:
            EventHandler creado
        
        Example:
            def handle_user_event(event):
                print(f"User event: {event.data}")
            
            bus.subscribe("user.created", handle_user_event)
        """
        if handler is None:
            # Usar como decorador
            def decorator(func):
                event_handler = EventHandler(func, filter_func, priority)
                if event_type:
                    with self._lock:
                        self._handlers[event_type].append(event_handler)
                        self._handlers[event_type].sort(key=lambda h: -h.priority)
                else:
                    with self._lock:
                        self._global_handlers.append(event_handler)
                        self._global_handlers.sort(key=lambda h: -h.priority)
                return func
            return decorator
        
        event_handler = EventHandler(handler, filter_func, priority)
        
        with self._lock:
            if event_type:
                self._handlers[event_type].append(event_handler)
                self._handlers[event_type].sort(key=lambda h: -h.priority)
            else:
                self._global_handlers.append(event_handler)
                self._global_handlers.sort(key=lambda h: -h.priority)
        
        return event_handler
    
    def unsubscribe(self, handler: EventHandler) -> bool:
        """
        Desuscribirse de eventos.
        
        Args:
            handler: Handler a remover
        
        Returns:
            True si se removió, False si no existía
        """
        with self._lock:
            # Remover de handlers específicos
            for handlers in self._handlers.values():
                if handler in handlers:
                    handlers.remove(handler)
                    return True
            
            # Remover de handlers globales
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
                return True
        
        return False
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        **metadata: Any
    ) -> None:
        """
        Publicar evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            priority: Prioridad del evento
            source: Fuente del evento (opcional)
            **metadata: Metadatos adicionales
        
        Example:
            await bus.publish("user.created", {"user_id": 123}, source="api")
        """
        event = Event(
            type=event_type,
            data=data,
            priority=priority,
            source=source,
            metadata=metadata
        )
        
        # Agregar a historial
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        
        # Obtener handlers
        with self._lock:
            handlers = self._handlers.get(event_type, []).copy()
            global_handlers = self._global_handlers.copy()
        
        # Procesar handlers específicos
        for handler in handlers:
            if handler.should_handle(event):
                try:
                    await handler.handle(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}", exc_info=True)
        
        # Procesar handlers globales
        for handler in global_handlers:
            if handler.should_handle(event):
                try:
                    await handler.handle(event)
                except Exception as e:
                    logger.error(f"Error in global event handler: {e}", exc_info=True)
    
    def publish_sync(
        self,
        event_type: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        **metadata: Any
    ) -> None:
        """
        Publicar evento de forma síncrona.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            priority: Prioridad del evento
            source: Fuente del evento (opcional)
            **metadata: Metadatos adicionales
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(
            self.publish(event_type, data, priority, source, **metadata)
        )
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Obtener historial de eventos.
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de eventos (opcional)
        
        Returns:
            Lista de eventos
        """
        with self._lock:
            events = self._event_history.copy()
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def clear_history(self) -> None:
        """Limpiar historial de eventos."""
        with self._lock:
            self._event_history.clear()


# Instancia global
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Obtener instancia global de EventBus."""
    return _event_bus


__all__ = [
    "EventPriority",
    "Event",
    "EventHandler",
    "EventBus",
    "get_event_bus",
]


"""
Sistema de eventos y hooks para Robot Movement AI v2.0
Event-driven architecture con handlers y callbacks
"""

from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from functools import wraps


class EventType(str, Enum):
    """Tipos de eventos disponibles"""
    ROBOT_CONNECTED = "robot.connected"
    ROBOT_DISCONNECTED = "robot.disconnected"
    MOVEMENT_STARTED = "movement.started"
    MOVEMENT_COMPLETED = "movement.completed"
    MOVEMENT_FAILED = "movement.failed"
    ERROR_OCCURRED = "error.occurred"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker.opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker.closed"


@dataclass
class Event:
    """Representa un evento"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir evento a diccionario"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


class EventEmitter:
    """Emisor de eventos con soporte para handlers async y sync"""
    
    def __init__(self):
        """Inicializar emisor de eventos"""
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        self.event_history: List[Event] = []
        self.max_history: int = 1000
    
    def on(self, event_type: EventType, handler: Callable):
        """
        Registrar handler para un tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Función handler (puede ser async o sync)
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def on_any(self, handler: Callable):
        """
        Registrar handler para todos los eventos
        
        Args:
            handler: Función handler
        """
        self.global_handlers.append(handler)
    
    def off(self, event_type: EventType, handler: Optional[Callable] = None):
        """
        Remover handler de un tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Handler específico a remover (None para remover todos)
        """
        if event_type in self.handlers:
            if handler is None:
                self.handlers[event_type].clear()
            elif handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
    
    async def emit(self, event: Event):
        """
        Emitir evento y ejecutar handlers
        
        Args:
            event: Evento a emitir
        """
        # Agregar a historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Ejecutar handlers específicos
        handlers = self.handlers.get(event.type, [])
        for handler in handlers:
            await self._execute_handler(handler, event)
        
        # Ejecutar handlers globales
        for handler in self.global_handlers:
            await self._execute_handler(handler, event)
    
    async def _execute_handler(self, handler: Callable, event: Event):
        """Ejecutar handler (async o sync)"""
        import inspect
        
        if inspect.iscoroutinefunction(handler):
            await handler(event)
        else:
            handler(event)
    
    def emit_sync(self, event: Event):
        """
        Emitir evento de forma síncrona (para casos simples)
        
        Args:
            event: Evento a emitir
        """
        # Agregar a historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Ejecutar handlers específicos
        handlers = self.handlers.get(event.type, [])
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                # Crear task para async handlers
                asyncio.create_task(handler(event))
            else:
                handler(event)
        
        # Ejecutar handlers globales
        for handler in self.global_handlers:
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler(event))
            else:
                handler(event)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """
        Obtener historial de eventos
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de eventos a retornar
            
        Returns:
            Lista de eventos
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return events[-limit:]
    
    def clear_history(self):
        """Limpiar historial de eventos"""
        self.event_history.clear()


# Instancia global de event emitter
_event_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Obtener instancia global de event emitter"""
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = EventEmitter()
    return _event_emitter


def emit_event(event_type: EventType, data: Dict[str, Any], source: Optional[str] = None):
    """Helper para emitir evento"""
    emitter = get_event_emitter()
    event = Event(type=event_type, data=data, source=source)
    emitter.emit_sync(event)


async def emit_event_async(event_type: EventType, data: Dict[str, Any], source: Optional[str] = None):
    """Helper para emitir evento async"""
    emitter = get_event_emitter()
    event = Event(type=event_type, data=data, source=source)
    await emitter.emit(event)


def on_event(event_type: EventType):
    """
    Decorator para registrar handler de evento
    
    Usage:
        @on_event(EventType.ROBOT_CONNECTED)
        async def handle_robot_connected(event: Event):
            print(f"Robot connected: {event.data}")
    """
    def decorator(func: Callable) -> Callable:
        emitter = get_event_emitter()
        emitter.on(event_type, func)
        return func
    return decorator

"""
Sistema de eventos y hooks
"""

from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio


class EventType(str, Enum):
    """Tipos de eventos"""
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    RECOMMENDATION_GENERATED = "recommendation.generated"
    ALERT_CREATED = "alert.created"
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    BACKUP_CREATED = "backup.created"
    WEBHOOK_SENT = "webhook.sent"
    NOTIFICATION_SENT = "notification.sent"


@dataclass
class Event:
    """Evento del sistema"""
    type: EventType
    data: Dict[str, Any]
    timestamp: str = None
    source: str = "system"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source
        }


class EventSystem:
    """Sistema de eventos y hooks"""
    
    def __init__(self):
        """Inicializa el sistema de eventos"""
        self.handlers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 10000
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Suscribe un handler a un evento
        
        Args:
            event_type: Tipo de evento
            handler: Función handler
        """
        event_key = event_type.value
        
        if event_key not in self.handlers:
            self.handlers[event_key] = []
        
        self.handlers[event_key].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """
        Desuscribe un handler de un evento
        
        Args:
            event_type: Tipo de evento
            handler: Función handler
        """
        event_key = event_type.value
        
        if event_key in self.handlers:
            if handler in self.handlers[event_key]:
                self.handlers[event_key].remove(handler)
    
    async def emit(self, event_type: EventType, data: Dict[str, Any],
                   source: str = "system"):
        """
        Emite un evento
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        event = Event(
            type=event_type,
            data=data,
            source=source
        )
        
        # Guardar en historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Ejecutar handlers
        event_key = event_type.value
        if event_key in self.handlers:
            tasks = []
            for handler in self.handlers[event_key]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error ejecutando handler: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[Event]:
        """
        Obtiene historial de eventos
        
        Args:
            event_type: Tipo de evento (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return events[-limit:]
    
    def get_event_stats(self) -> Dict:
        """Obtiene estadísticas de eventos"""
        stats = {}
        
        for event in self.event_history:
            event_type = event.type.value
            stats[event_type] = stats.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.event_history),
            "by_type": stats,
            "handlers": {
                event_type: len(handlers)
                for event_type, handlers in self.handlers.items()
            }
        }







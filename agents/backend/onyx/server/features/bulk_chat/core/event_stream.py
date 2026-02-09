"""
Event Stream - Flujo de Eventos en Tiempo Real
==============================================

Sistema de gestión de eventos en tiempo real con pub/sub, filtrado y procesamiento de streams.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipo de evento."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    METRIC = "metric"
    ALERT = "alert"
    CUSTOM = "custom"


@dataclass
class Event:
    """Evento."""
    event_id: str
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSubscription:
    """Suscripción a eventos."""
    subscription_id: str
    event_types: List[EventType]
    filters: Dict[str, Any] = field(default_factory=dict)
    handler: Callable = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class EventStream:
    """Sistema de flujo de eventos."""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.event_history: deque = deque(maxlen=history_size)
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def publish(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publicar evento."""
        event_id = f"event_{source}_{datetime.now().timestamp()}"
        
        event = Event(
            event_id=event_id,
            event_type=event_type,
            source=source,
            data=data,
            metadata=metadata or {},
        )
        
        self.event_history.append(event)
        
        # Notificar a handlers
        asyncio.create_task(self._notify_handlers(event))
        
        logger.debug(f"Published event: {event_id} ({event_type.value})")
        return event_id
    
    async def _notify_handlers(self, event: Event):
        """Notificar a handlers."""
        # Handlers específicos por tipo
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        # Handlers de suscripciones
        for subscription in self.subscriptions.values():
            if not subscription.active:
                continue
            
            if event.event_type not in subscription.event_types:
                continue
            
            # Aplicar filtros
            if self._matches_filters(event, subscription.filters):
                try:
                    if asyncio.iscoroutinefunction(subscription.handler):
                        await subscription.handler(event)
                    else:
                        subscription.handler(event)
                except Exception as e:
                    logger.error(f"Error in subscription handler: {e}")
    
    def _matches_filters(self, event: Event, filters: Dict[str, Any]) -> bool:
        """Verificar si evento coincide con filtros."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == "source":
                if event.source != value:
                    return False
            elif key == "data":
                # Filtro en data
                for data_key, data_value in value.items():
                    if event.data.get(data_key) != data_value:
                        return False
            elif key == "metadata":
                # Filtro en metadata
                for meta_key, meta_value in value.items():
                    if event.metadata.get(meta_key) != meta_value:
                        return False
        
        return True
    
    def subscribe(
        self,
        subscription_id: str,
        event_types: List[EventType],
        handler: Callable,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Suscribirse a eventos."""
        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_types=event_types,
            filters=filters or {},
            handler=handler,
        )
        
        async def save_subscription():
            async with self._lock:
                self.subscriptions[subscription_id] = subscription
        
        asyncio.create_task(save_subscription())
        
        logger.info(f"Subscribed to events: {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Desuscribirse de eventos."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return False
        
        subscription.active = False
        return True
    
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable,
    ):
        """Registrar handler para tipo de evento."""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener eventos."""
        events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "source": e.source,
                "data": e.data,
                "timestamp": e.timestamp.isoformat(),
                "metadata": e.metadata,
            }
            for e in events[:limit]
        ]
    
    def get_event_stream_summary(self) -> Dict[str, Any]:
        """Obtener resumen del stream."""
        by_type: Dict[str, int] = defaultdict(int)
        by_source: Dict[str, int] = defaultdict(int)
        
        for event in self.event_history:
            by_type[event.event_type.value] += 1
            by_source[event.source] += 1
        
        return {
            "total_events": len(self.event_history),
            "events_by_type": dict(by_type),
            "events_by_source": dict(by_source),
            "total_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "total_handlers": sum(len(handlers) for handlers in self.event_handlers.values()),
        }















"""
Event Bus - Bus de Eventos Distribuido
========================================

Sistema de bus de eventos con pub/sub, filtrado y procesamiento asíncrono.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Prioridad de evento."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Evento."""
    event_id: str
    event_type: str
    source: str
    payload: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """Suscripción."""
    subscription_id: str
    subscriber_id: str
    event_types: List[str]
    handler: Callable
    filter_func: Optional[Callable] = None
    priority: EventPriority = EventPriority.NORMAL
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Bus de eventos distribuido."""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=100000)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._processing_active = False
    
    def start_processing(self):
        """Iniciar procesamiento de eventos."""
        if self._processing_active:
            return
        
        self._processing_active = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus processing started")
    
    def stop_processing(self):
        """Detener procesamiento de eventos."""
        self._processing_active = False
        
        if self._processing_task:
            self._processing_task.cancel()
            self._processing_task = None
        
        logger.info("Event bus processing stopped")
    
    async def _process_events(self):
        """Procesar eventos de la cola."""
        while self._processing_active:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def publish(
        self,
        event_type: str,
        source: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publicar evento."""
        event = Event(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            source=source,
            payload=payload,
            priority=priority,
            metadata=metadata or {},
        )
        
        # Agregar a historial
        async with self._lock:
            self.event_history.append(event)
        
        # Agregar a cola
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.event_id}")
        
        logger.debug(f"Published event: {event.event_id} - {event_type}")
        return event.event_id
    
    async def _dispatch_event(self, event: Event):
        """Despachar evento a suscriptores."""
        # Obtener suscripciones para este tipo de evento
        subscriptions = self.subscriptions.get(event.event_type, [])
        
        # Filtrar por habilitadas y que cumplan filtro
        valid_subscriptions = [
            sub for sub in subscriptions
            if sub.enabled and (not sub.filter_func or sub.filter_func(event))
        ]
        
        # Ordenar por prioridad
        valid_subscriptions.sort(key=lambda s: s.priority.value, reverse=True)
        
        # Despachar a todos los suscriptores
        for subscription in valid_subscriptions:
            try:
                if asyncio.iscoroutinefunction(subscription.handler):
                    await subscription.handler(event)
                else:
                    subscription.handler(event)
            except Exception as e:
                logger.error(f"Error in subscription handler {subscription.subscription_id}: {e}")
    
    def subscribe(
        self,
        subscription_id: str,
        subscriber_id: str,
        event_types: List[str],
        handler: Callable,
        filter_func: Optional[Callable] = None,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Suscribirse a eventos."""
        subscription = Subscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_types=event_types,
            handler=handler,
            filter_func=filter_func,
            priority=priority,
            metadata=metadata or {},
        )
        
        async def save_subscription():
            async with self._lock:
                for event_type in event_types:
                    self.subscriptions[event_type].append(subscription)
        
        asyncio.create_task(save_subscription())
        
        logger.info(f"Subscribed: {subscription_id} to {event_types}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Desuscribirse de eventos."""
        async with self._lock:
            removed = False
            for event_type, subscriptions in self.subscriptions.items():
                self.subscriptions[event_type] = [
                    s for s in subscriptions if s.subscription_id != subscription_id
                ]
                if len(self.subscriptions[event_type]) < len(subscriptions):
                    removed = True
        
        if removed:
            logger.info(f"Unsubscribed: {subscription_id}")
        
        return removed
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de eventos."""
        history = list(self.event_history)
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        
        if source:
            history = [e for e in history if e.source == source]
        
        history.sort(key=lambda e: e.timestamp, reverse=True)
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "source": e.source,
                "payload": e.payload,
                "priority": e.priority.value,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in history[:limit]
        ]
    
    def get_subscriptions(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener suscripciones."""
        all_subscriptions = []
        
        if event_type:
            subscriptions = self.subscriptions.get(event_type, [])
        else:
            subscriptions = []
            for subs in self.subscriptions.values():
                subscriptions.extend(subs)
        
        # Remover duplicados
        seen = set()
        for sub in subscriptions:
            if sub.subscription_id not in seen:
                seen.add(sub.subscription_id)
                all_subscriptions.append({
                    "subscription_id": sub.subscription_id,
                    "subscriber_id": sub.subscriber_id,
                    "event_types": sub.event_types,
                    "priority": sub.priority.value,
                    "enabled": sub.enabled,
                })
        
        return all_subscriptions
    
    def get_event_bus_summary(self) -> Dict[str, Any]:
        """Obtener resumen del bus."""
        by_type: Dict[str, int] = defaultdict(int)
        by_priority: Dict[str, int] = defaultdict(int)
        
        for event in self.event_history:
            by_type[event.event_type] += 1
            by_priority[event.priority.value] += 1
        
        return {
            "processing_active": self._processing_active,
            "queue_size": self.event_queue.qsize(),
            "total_events": len(self.event_history),
            "events_by_type": dict(by_type),
            "events_by_priority": dict(by_priority),
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "event_types": list(self.subscriptions.keys()),
        }



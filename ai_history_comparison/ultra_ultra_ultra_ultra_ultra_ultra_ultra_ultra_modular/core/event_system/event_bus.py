"""
Event Bus - Bus de Eventos
=========================

Sistema de eventos distribuido con patrones avanzados de event sourcing y CQRS.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Type, Set
from datetime import datetime
from enum import Enum
import json
import uuid

from ..interfaces.base_interfaces import IEventBus, IEvent, IObserver, ISubject, IEventStore

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos."""
    DOMAIN = "domain"
    INTEGRATION = "integration"
    SYSTEM = "system"
    USER = "user"
    AUDIT = "audit"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Prioridades de eventos."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Estados de eventos."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class BaseEvent(IEvent):
    """Evento base."""
    
    def __init__(self, event_type: str, data: Any, metadata: Dict[str, Any] = None):
        self._event_id = str(uuid.uuid4())
        self._event_type = event_type
        self._timestamp = datetime.utcnow()
        self._data = data
        self._metadata = metadata or {}
        self._status = EventStatus.PENDING
        self._priority = EventPriority.NORMAL
        self._retry_count = 0
        self._max_retries = 3
        self._correlation_id = None
        self._causation_id = None
    
    @property
    def event_id(self) -> str:
        return self._event_id
    
    @property
    def event_type(self) -> str:
        return self._event_type
    
    @property
    def timestamp(self) -> datetime:
        return self._timestamp
    
    @property
    def data(self) -> Any:
        return self._data
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    @property
    def status(self) -> EventStatus:
        return self._status
    
    @status.setter
    def status(self, value: EventStatus):
        self._status = value
    
    @property
    def priority(self) -> EventPriority:
        return self._priority
    
    @priority.setter
    def priority(self, value: EventPriority):
        self._priority = value
    
    @property
    def retry_count(self) -> int:
        return self._retry_count
    
    @retry_count.setter
    def retry_count(self, value: int):
        self._retry_count = value
    
    @property
    def max_retries(self) -> int:
        return self._max_retries
    
    @max_retries.setter
    def max_retries(self, value: int):
        self._max_retries = value
    
    @property
    def correlation_id(self) -> Optional[str]:
        return self._correlation_id
    
    @correlation_id.setter
    def correlation_id(self, value: Optional[str]):
        self._correlation_id = value
    
    @property
    def causation_id(self) -> Optional[str]:
        return self._causation_id
    
    @causation_id.setter
    def causation_id(self, value: Optional[str]):
        self._causation_id = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir evento a diccionario."""
        return {
            "event_id": self._event_id,
            "event_type": self._event_type,
            "timestamp": self._timestamp.isoformat(),
            "data": self._data,
            "metadata": self._metadata,
            "status": self._status.value,
            "priority": self._priority.value,
            "retry_count": self._retry_count,
            "max_retries": self._max_retries,
            "correlation_id": self._correlation_id,
            "causation_id": self._causation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Crear evento desde diccionario."""
        event = cls(data["event_type"], data["data"], data.get("metadata", {}))
        event._event_id = data["event_id"]
        event._timestamp = datetime.fromisoformat(data["timestamp"])
        event._status = EventStatus(data["status"])
        event._priority = EventPriority(data["priority"])
        event._retry_count = data["retry_count"]
        event._max_retries = data["max_retries"]
        event._correlation_id = data.get("correlation_id")
        event._causation_id = data.get("causation_id")
        return event


class EventHandler:
    """Manejador de eventos."""
    
    def __init__(self, handler: Callable, event_types: List[str] = None, 
                 priority: int = 0, async_handler: bool = True):
        self.handler = handler
        self.event_types = event_types or []
        self.priority = priority
        self.async_handler = async_handler
        self.handler_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.execution_count = 0
        self.last_execution = None
        self.total_execution_time = 0.0
        self.error_count = 0
    
    async def handle(self, event: BaseEvent) -> Any:
        """Manejar evento."""
        start_time = datetime.utcnow()
        
        try:
            if self.async_handler:
                result = await self.handler(event)
            else:
                result = self.handler(event)
            
            # Actualizar estadísticas
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.execution_count += 1
            self.last_execution = datetime.utcnow()
            self.total_execution_time += execution_time
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in event handler {self.handler_id}: {e}")
            raise
    
    def can_handle(self, event: BaseEvent) -> bool:
        """Verificar si puede manejar el evento."""
        if not self.event_types:
            return True
        return event.event_type in self.event_types
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del manejador."""
        avg_execution_time = (self.total_execution_time / self.execution_count 
                             if self.execution_count > 0 else 0.0)
        
        return {
            "handler_id": self.handler_id,
            "event_types": self.event_types,
            "priority": self.priority,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "created_at": self.created_at.isoformat()
        }


class EventSubscription:
    """Suscripción a eventos."""
    
    def __init__(self, subscription_id: str, event_types: List[str], 
                 handler: EventHandler, filters: Dict[str, Any] = None):
        self.subscription_id = subscription_id
        self.event_types = event_types
        self.handler = handler
        self.filters = filters or {}
        self.created_at = datetime.utcnow()
        self.active = True
        self.event_count = 0
    
    def matches(self, event: BaseEvent) -> bool:
        """Verificar si la suscripción coincide con el evento."""
        if not self.active:
            return False
        
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Aplicar filtros
        for key, value in self.filters.items():
            if key in event.metadata:
                if event.metadata[key] != value:
                    return False
            elif key == "priority":
                if event.priority.value != value:
                    return False
            elif key == "status":
                if event.status.value != value:
                    return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la suscripción."""
        return {
            "subscription_id": self.subscription_id,
            "event_types": self.event_types,
            "filters": self.filters,
            "active": self.active,
            "event_count": self.event_count,
            "created_at": self.created_at.isoformat(),
            "handler_stats": self.handler.get_stats()
        }


class EventBus(IEventBus, IComponent):
    """Bus de eventos distribuido."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._max_workers = 10
        self._event_store: Optional[IEventStore] = None
        self._event_stats: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Inicializar bus de eventos."""
        try:
            # Iniciar workers de procesamiento
            for i in range(self._max_workers):
                task = asyncio.create_task(self._event_worker(f"worker-{i}"))
                self._processing_tasks.add(task)
            
            self._initialized = True
            logger.info(f"Event bus {self.name} initialized with {self._max_workers} workers")
            
        except Exception as e:
            logger.error(f"Error initializing event bus: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cerrar bus de eventos."""
        try:
            # Señalar shutdown
            self._shutdown_event.set()
            
            # Cancelar workers
            for task in self._processing_tasks:
                task.cancel()
            
            # Esperar a que terminen
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            
            self._processing_tasks.clear()
            self._initialized = False
            logger.info(f"Event bus {self.name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down event bus: {e}")
    
    async def health_check(self) -> bool:
        """Verificar salud del bus."""
        return self._initialized and not self._shutdown_event.is_set()
    
    @property
    def name(self) -> str:
        return f"EventBus_{self.name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def publish(self, topic: str, message: Any) -> bool:
        """Publicar mensaje."""
        try:
            # Crear evento
            event = BaseEvent(topic, message)
            
            # Agregar a cola
            await self._event_queue.put(event)
            
            # Almacenar en event store si está disponible
            if self._event_store:
                await self._event_store.append(event)
            
            # Actualizar estadísticas
            await self._update_event_stats("published", event)
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event to topic {topic}: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: callable) -> str:
        """Suscribirse a tópico."""
        try:
            subscription_id = str(uuid.uuid4())
            event_handler = EventHandler(handler)
            subscription = EventSubscription(
                subscription_id, 
                [topic], 
                event_handler
            )
            
            async with self._lock:
                self._subscriptions[subscription_id] = subscription
            
            logger.info(f"Subscribed to topic {topic} with ID {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Cancelar suscripción."""
        try:
            async with self._lock:
                if subscription_id in self._subscriptions:
                    subscription = self._subscriptions[subscription_id]
                    subscription.active = False
                    del self._subscriptions[subscription_id]
                    logger.info(f"Unsubscribed {subscription_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error unsubscribing {subscription_id}: {e}")
            return False
    
    async def emit(self, event_type: str, event_data: Any) -> bool:
        """Emitir evento."""
        return await self.publish(event_type, event_data)
    
    async def listen(self, event_type: str, handler: callable) -> str:
        """Escuchar evento."""
        return await self.subscribe(event_type, handler)
    
    async def _event_worker(self, worker_name: str) -> None:
        """Worker para procesar eventos."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Obtener evento de la cola con timeout
                    event = await asyncio.wait_for(
                        self._event_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Procesar evento
                    await self._process_event(event, worker_name)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in event worker {worker_name}: {e}")
                    
        except asyncio.CancelledError:
            logger.info(f"Event worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Fatal error in event worker {worker_name}: {e}")
    
    async def _process_event(self, event: BaseEvent, worker_name: str) -> None:
        """Procesar evento individual."""
        try:
            event.status = EventStatus.PROCESSING
            
            # Encontrar suscripciones que coincidan
            matching_subscriptions = []
            async with self._lock:
                for subscription in self._subscriptions.values():
                    if subscription.matches(event):
                        matching_subscriptions.append(subscription)
            
            # Ordenar por prioridad del handler
            matching_subscriptions.sort(
                key=lambda s: s.handler.priority, 
                reverse=True
            )
            
            # Ejecutar handlers
            for subscription in matching_subscriptions:
                try:
                    await subscription.handler.handle(event)
                    subscription.event_count += 1
                    
                except Exception as e:
                    logger.error(f"Error in event handler {subscription.handler.handler_id}: {e}")
                    
                    # Manejar reintentos
                    if event.retry_count < event.max_retries:
                        event.retry_count += 1
                        event.status = EventStatus.RETRYING
                        # Reagregar a la cola para reintento
                        await self._event_queue.put(event)
                        return
                    else:
                        event.status = EventStatus.FAILED
                        await self._update_event_stats("failed", event)
                        return
            
            # Marcar como completado
            event.status = EventStatus.COMPLETED
            await self._update_event_stats("completed", event)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            event.status = EventStatus.FAILED
            await self._update_event_stats("failed", event)
    
    async def _update_event_stats(self, action: str, event: BaseEvent) -> None:
        """Actualizar estadísticas de eventos."""
        try:
            if action not in self._event_stats:
                self._event_stats[action] = 0
            self._event_stats[action] += 1
            
            # Estadísticas por tipo de evento
            event_type_key = f"{action}_{event.event_type}"
            if event_type_key not in self._event_stats:
                self._event_stats[event_type_key] = 0
            self._event_stats[event_type_key] += 1
            
        except Exception as e:
            logger.error(f"Error updating event stats: {e}")
    
    async def set_event_store(self, event_store: IEventStore) -> None:
        """Establecer event store."""
        self._event_store = event_store
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de eventos."""
        try:
            subscription_stats = []
            async with self._lock:
                for subscription in self._subscriptions.values():
                    subscription_stats.append(subscription.get_stats())
            
            return {
                "bus_name": self.name,
                "initialized": self._initialized,
                "worker_count": len(self._processing_tasks),
                "queue_size": self._event_queue.qsize(),
                "subscription_count": len(self._subscriptions),
                "event_stats": self._event_stats.copy(),
                "subscription_stats": subscription_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting event stats: {e}")
            return {}
    
    async def get_subscription_stats(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas de suscripción específica."""
        async with self._lock:
            subscription = self._subscriptions.get(subscription_id)
            if subscription:
                return subscription.get_stats()
            return None
    
    async def pause_subscription(self, subscription_id: str) -> bool:
        """Pausar suscripción."""
        try:
            async with self._lock:
                subscription = self._subscriptions.get(subscription_id)
                if subscription:
                    subscription.active = False
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error pausing subscription {subscription_id}: {e}")
            return False
    
    async def resume_subscription(self, subscription_id: str) -> bool:
        """Reanudar suscripción."""
        try:
            async with self._lock:
                subscription = self._subscriptions.get(subscription_id)
                if subscription:
                    subscription.active = True
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error resuming subscription {subscription_id}: {e}")
            return False
    
    async def clear_event_queue(self) -> int:
        """Limpiar cola de eventos."""
        try:
            cleared_count = 0
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                    cleared_count += 1
                except asyncio.QueueEmpty:
                    break
            
            logger.info(f"Cleared {cleared_count} events from queue")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing event queue: {e}")
            return 0
    
    async def get_queue_size(self) -> int:
        """Obtener tamaño de la cola."""
        return self._event_queue.qsize()
    
    async def set_max_workers(self, max_workers: int) -> bool:
        """Establecer número máximo de workers."""
        try:
            if max_workers <= 0:
                return False
            
            old_workers = self._max_workers
            self._max_workers = max_workers
            
            if max_workers > old_workers:
                # Agregar workers
                for i in range(old_workers, max_workers):
                    task = asyncio.create_task(self._event_worker(f"worker-{i}"))
                    self._processing_tasks.add(task)
            elif max_workers < old_workers:
                # Remover workers
                tasks_to_remove = list(self._processing_tasks)[max_workers:]
                for task in tasks_to_remove:
                    task.cancel()
                    self._processing_tasks.discard(task)
            
            logger.info(f"Updated max workers from {old_workers} to {max_workers}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting max workers: {e}")
            return False





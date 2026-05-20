"""
Event System - Sistema de eventos para comunicación entre agentes
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos del sistema."""
    # Eventos de agentes
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    
    # Eventos de tareas
    TASK_SUBMITTED = "task_submitted"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    
    # Eventos de flujos de trabajo
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    
    # Eventos del sistema
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    HEALTH_CHECK = "health_check"
    METRICS_UPDATE = "metrics_update"
    
    # Eventos de comunicación
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    BROADCAST = "broadcast"


@dataclass
class Event:
    """Evento del sistema."""
    event_id: str
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1-10, mayor número = mayor prioridad
    ttl: int = 3600  # Tiempo de vida en segundos


@dataclass
class EventHandler:
    """Manejador de eventos."""
    handler_id: str
    event_types: List[EventType]
    handler: Callable
    async_handler: bool = False
    priority: int = 1
    active: bool = True


class EventSystem:
    """
    Sistema de eventos para comunicación entre agentes.
    """
    
    def __init__(self):
        """Inicializar el sistema de eventos."""
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Event] = []
        self.running = False
        
        # Configuración
        self.max_history_size = 10000
        self.event_ttl = 3600  # 1 hora
        self.max_queue_size = 1000
        
        # Métricas
        self.metrics = {
            "total_events": 0,
            "events_processed": 0,
            "events_failed": 0,
            "active_handlers": 0,
            "queue_size": 0
        }
        
        logger.info("EventSystem inicializado")
    
    async def initialize(self):
        """Inicializar el sistema de eventos."""
        try:
            self.running = True
            
            # Iniciar procesador de eventos
            asyncio.create_task(self._event_processor())
            
            # Iniciar limpiador de eventos
            asyncio.create_task(self._event_cleaner())
            
            logger.info("EventSystem inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar EventSystem: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el sistema de eventos."""
        try:
            self.running = False
            
            # Procesar eventos restantes
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                    await self._process_event(event)
                except asyncio.QueueEmpty:
                    break
            
            logger.info("EventSystem cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar EventSystem: {e}")
    
    async def emit_event(
        self,
        event_type: Union[EventType, str],
        data: Dict[str, Any],
        source: str = "system",
        priority: int = 1,
        ttl: int = None
    ) -> str:
        """
        Emitir un evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            priority: Prioridad del evento (1-10)
            ttl: Tiempo de vida del evento
            
        Returns:
            ID del evento emitido
        """
        try:
            # Convertir string a EventType si es necesario
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    logger.warning(f"Tipo de evento desconocido: {event_type}")
                    return None
            
            # Generar ID único
            import uuid
            event_id = str(uuid.uuid4())
            
            # Crear evento
            event = Event(
                event_id=event_id,
                event_type=event_type,
                source=source,
                data=data,
                priority=priority,
                ttl=ttl or self.event_ttl
            )
            
            # Agregar a la cola
            if self.event_queue.qsize() < self.max_queue_size:
                await self.event_queue.put(event)
                self.metrics["total_events"] += 1
                self.metrics["queue_size"] = self.event_queue.qsize()
                
                logger.debug(f"Evento {event_id} emitido: {event_type.value}")
                return event_id
            else:
                logger.warning("Cola de eventos llena, evento descartado")
                return None
                
        except Exception as e:
            logger.error(f"Error al emitir evento: {e}")
            return None
    
    async def subscribe(
        self,
        event_types: Union[EventType, List[EventType]],
        handler: Callable,
        handler_id: str = None,
        priority: int = 1
    ) -> str:
        """
        Suscribirse a eventos.
        
        Args:
            event_types: Tipos de eventos a escuchar
            handler: Función manejadora
            handler_id: ID único del manejador
            priority: Prioridad del manejador
            
        Returns:
            ID del manejador
        """
        try:
            # Generar ID si no se proporciona
            if not handler_id:
                import uuid
                handler_id = str(uuid.uuid4())
            
            # Normalizar tipos de eventos
            if isinstance(event_types, EventType):
                event_types = [event_types]
            
            # Verificar si es función asíncrona
            async_handler = asyncio.iscoroutinefunction(handler)
            
            # Crear manejador
            event_handler = EventHandler(
                handler_id=handler_id,
                event_types=event_types,
                handler=handler,
                async_handler=async_handler,
                priority=priority
            )
            
            # Registrar manejador
            for event_type in event_types:
                if event_type not in self.handlers:
                    self.handlers[event_type] = []
                
                self.handlers[event_type].append(event_handler)
                # Ordenar por prioridad
                self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
            
            self.metrics["active_handlers"] += 1
            
            logger.info(f"Manejador {handler_id} suscrito a {len(event_types)} tipos de eventos")
            return handler_id
            
        except Exception as e:
            logger.error(f"Error al suscribirse a eventos: {e}")
            return None
    
    async def unsubscribe(self, handler_id: str) -> bool:
        """
        Desuscribirse de eventos.
        
        Args:
            handler_id: ID del manejador
            
        Returns:
            True si se desuscribió exitosamente
        """
        try:
            removed_count = 0
            
            for event_type, handlers in self.handlers.items():
                handlers[:] = [h for h in handlers if h.handler_id != handler_id]
                removed_count += len([h for h in handlers if h.handler_id == handler_id])
            
            if removed_count > 0:
                self.metrics["active_handlers"] -= removed_count
                logger.info(f"Manejador {handler_id} desuscrito de {removed_count} tipos de eventos")
                return True
            else:
                logger.warning(f"Manejador {handler_id} no encontrado")
                return False
                
        except Exception as e:
            logger.error(f"Error al desuscribirse de eventos: {e}")
            return False
    
    async def _event_processor(self):
        """Procesador de eventos."""
        while self.running:
            try:
                # Obtener evento de la cola
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Procesar evento
                await self._process_event(event)
                
                # Agregar al historial
                self._add_to_history(event)
                
                # Actualizar métricas
                self.metrics["events_processed"] += 1
                self.metrics["queue_size"] = self.event_queue.qsize()
                
            except asyncio.TimeoutError:
                # Timeout normal, continuar
                continue
            except Exception as e:
                logger.error(f"Error en procesador de eventos: {e}")
                self.metrics["events_failed"] += 1
    
    async def _process_event(self, event: Event):
        """Procesar un evento individual."""
        try:
            # Obtener manejadores para este tipo de evento
            handlers = self.handlers.get(event.event_type, [])
            
            if not handlers:
                logger.debug(f"No hay manejadores para evento {event.event_type.value}")
                return
            
            # Ejecutar manejadores
            for handler in handlers:
                if not handler.active:
                    continue
                
                try:
                    if handler.async_handler:
                        await handler.handler(event)
                    else:
                        handler.handler(event)
                except Exception as e:
                    logger.error(f"Error en manejador {handler.handler_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error al procesar evento {event.event_id}: {e}")
    
    def _add_to_history(self, event: Event):
        """Agregar evento al historial."""
        self.event_history.append(event)
        
        # Limpiar historial si es muy grande
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    async def _event_cleaner(self):
        """Limpiador de eventos expirados."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Ejecutar cada minuto
                
                current_time = datetime.now()
                expired_events = []
                
                # Encontrar eventos expirados
                for event in self.event_history:
                    if (current_time - event.timestamp).total_seconds() > event.ttl:
                        expired_events.append(event)
                
                # Remover eventos expirados
                for event in expired_events:
                    self.event_history.remove(event)
                
                if expired_events:
                    logger.debug(f"Removidos {len(expired_events)} eventos expirados")
                    
            except Exception as e:
                logger.error(f"Error en limpiador de eventos: {e}")
    
    async def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de eventos.
        
        Args:
            event_type: Filtrar por tipo de evento
            source: Filtrar por fuente
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        try:
            filtered_events = self.event_history
            
            # Aplicar filtros
            if event_type:
                filtered_events = [e for e in filtered_events if e.event_type == event_type]
            
            if source:
                filtered_events = [e for e in filtered_events if e.source == source]
            
            # Limitar resultados
            filtered_events = filtered_events[-limit:]
            
            # Convertir a diccionarios
            return [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "source": event.source,
                    "data": event.data,
                    "timestamp": event.timestamp.isoformat(),
                    "priority": event.priority,
                    "ttl": event.ttl
                }
                for event in filtered_events
            ]
            
        except Exception as e:
            logger.error(f"Error al obtener historial de eventos: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema de eventos."""
        return {
            **self.metrics,
            "queue_size": self.event_queue.qsize(),
            "history_size": len(self.event_history),
            "handler_types": len(self.handlers),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de eventos."""
        try:
            return {
                "status": "healthy",
                "running": self.running,
                "queue_size": self.event_queue.qsize(),
                "history_size": len(self.event_history),
                "active_handlers": self.metrics["active_handlers"],
                "metrics": self.metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check del EventSystem: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def broadcast_message(
        self,
        message: str,
        data: Dict[str, Any] = None,
        source: str = "system"
    ) -> str:
        """
        Enviar mensaje de difusión.
        
        Args:
            message: Mensaje a difundir
            data: Datos adicionales
            source: Fuente del mensaje
            
        Returns:
            ID del evento
        """
        return await self.emit_event(
            event_type=EventType.BROADCAST,
            data={
                "message": message,
                "data": data or {}
            },
            source=source,
            priority=5
        )
    
    async def send_message(
        self,
        target: str,
        message: str,
        data: Dict[str, Any] = None,
        source: str = "system"
    ) -> str:
        """
        Enviar mensaje a un agente específico.
        
        Args:
            target: Destinatario del mensaje
            message: Mensaje
            data: Datos adicionales
            source: Fuente del mensaje
            
        Returns:
            ID del evento
        """
        return await self.emit_event(
            event_type=EventType.MESSAGE_SENT,
            data={
                "target": target,
                "message": message,
                "data": data or {}
            },
            source=source,
            priority=3
        )





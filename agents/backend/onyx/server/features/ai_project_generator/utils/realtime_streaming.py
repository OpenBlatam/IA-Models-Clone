"""
Realtime Streaming - Streaming en Tiempo Real
=============================================

Sistema de streaming en tiempo real para actualizaciones de proyectos.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Tipos de eventos de streaming"""
    PROJECT_STARTED = "project.started"
    PROJECT_PROGRESS = "project.progress"
    PROJECT_COMPLETED = "project.completed"
    PROJECT_FAILED = "project.failed"
    QUEUE_UPDATED = "queue.updated"
    STATS_UPDATED = "stats.updated"
    SYSTEM_EVENT = "system.event"


class StreamEvent:
    """Evento de streaming"""
    
    def __init__(
        self,
        event_type: StreamEventType,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte evento a diccionario"""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_json(self) -> str:
        """Convierte evento a JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StreamManager:
    """Gestor de streams en tiempo real"""
    
    def __init__(self):
        """Inicializa el gestor de streams"""
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self.event_history: List[StreamEvent] = []
        self.max_history = 1000
        self.lock = asyncio.Lock()
    
    async def subscribe(
        self,
        event_type: StreamEventType,
        callback: Callable,
    ):
        """
        Suscribe un callback a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        async with self.lock:
            self.subscribers[event_type.value].add(callback)
        logger.info(f"Callback suscrito a evento: {event_type.value}")
    
    async def unsubscribe(
        self,
        event_type: StreamEventType,
        callback: Callable,
    ):
        """Desuscribe un callback"""
        async with self.lock:
            self.subscribers[event_type.value].discard(callback)
    
    async def emit(
        self,
        event_type: StreamEventType,
        data: Dict[str, Any],
    ):
        """
        Emite un evento a todos los suscriptores.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        event = StreamEvent(event_type, data)
        
        # Guardar en historial
        async with self.lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        # Notificar a suscriptores
        callbacks = list(self.subscribers.get(event_type.value, set()))
        if callbacks:
            tasks = []
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(callback(event))
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error en callback de evento {event_type.value}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Evento emitido: {event_type.value}")
    
    def get_event_history(
        self,
        event_type: Optional[StreamEventType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Obtiene historial de eventos.
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return [e.to_dict() for e in events[-limit:]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del stream manager"""
        return {
            "total_events": len(self.event_history),
            "subscribers_by_type": {
                event_type: len(callbacks)
                for event_type, callbacks in self.subscribers.items()
            },
            "event_types": [e.event_type.value for e in self.event_history[-10:]],
        }


class ProjectStreamer:
    """Streamer para eventos de proyectos"""
    
    def __init__(self, stream_manager: StreamManager):
        """Inicializa el streamer de proyectos"""
        self.stream_manager = stream_manager
    
    async def stream_project_started(
        self,
        project_id: str,
        description: str,
    ):
        """Emite evento de proyecto iniciado"""
        await self.stream_manager.emit(
            StreamEventType.PROJECT_STARTED,
            {
                "project_id": project_id,
                "description": description,
            }
        )
    
    async def stream_project_progress(
        self,
        project_id: str,
        progress: float,
        message: str,
    ):
        """Emite evento de progreso"""
        await self.stream_manager.emit(
            StreamEventType.PROJECT_PROGRESS,
            {
                "project_id": project_id,
                "progress": progress,
                "message": message,
            }
        )
    
    async def stream_project_completed(
        self,
        project_id: str,
        project_info: Dict[str, Any],
    ):
        """Emite evento de proyecto completado"""
        await self.stream_manager.emit(
            StreamEventType.PROJECT_COMPLETED,
            {
                "project_id": project_id,
                "project_info": project_info,
            }
        )
    
    async def stream_project_failed(
        self,
        project_id: str,
        error: str,
    ):
        """Emite evento de proyecto fallido"""
        await self.stream_manager.emit(
            StreamEventType.PROJECT_FAILED,
            {
                "project_id": project_id,
                "error": error,
            }
        )


class QueueStreamer:
    """Streamer para eventos de cola"""
    
    def __init__(self, stream_manager: StreamManager):
        """Inicializa el streamer de cola"""
        self.stream_manager = stream_manager
    
    async def stream_queue_updated(
        self,
        queue_size: int,
        queue_info: Dict[str, Any],
    ):
        """Emite evento de actualización de cola"""
        await self.stream_manager.emit(
            StreamEventType.QUEUE_UPDATED,
            {
                "queue_size": queue_size,
                "queue_info": queue_info,
            }
        )


class StatsStreamer:
    """Streamer para estadísticas"""
    
    def __init__(self, stream_manager: StreamManager):
        """Inicializa el streamer de estadísticas"""
        self.stream_manager = stream_manager
    
    async def stream_stats_updated(
        self,
        stats: Dict[str, Any],
    ):
        """Emite evento de actualización de estadísticas"""
        await self.stream_manager.emit(
            StreamEventType.STATS_UPDATED,
            {
                "stats": stats,
            }
        )


# Instancia global
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Obtiene la instancia global del stream manager"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager

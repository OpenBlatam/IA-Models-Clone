"""
Streaming System
================

Sistema de streaming de datos en tiempo real.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Estado de stream."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Stream:
    """Stream."""
    stream_id: str
    name: str
    description: str
    source: str  # Fuente de datos
    status: StreamStatus = StreamStatus.ACTIVE
    subscribers: List[str] = field(default_factory=list)  # IDs de suscriptores
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_data_at: Optional[str] = None


class StreamingSystem:
    """
    Sistema de streaming.
    
    Gestiona streams de datos en tiempo real.
    """
    
    def __init__(self):
        """Inicializar sistema de streaming."""
        self.streams: Dict[str, Stream] = {}
        self.stream_queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}  # stream_id -> queues
    
    def create_stream(
        self,
        stream_id: str,
        name: str,
        description: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Stream:
        """
        Crear stream.
        
        Args:
            stream_id: ID único del stream
            name: Nombre
            description: Descripción
            source: Fuente de datos
            metadata: Metadata adicional
            
        Returns:
            Stream creado
        """
        stream = Stream(
            stream_id=stream_id,
            name=name,
            description=description,
            source=source,
            metadata=metadata or {}
        )
        
        self.streams[stream_id] = stream
        self.stream_queues[stream_id] = asyncio.Queue()
        self.subscribers[stream_id] = []
        
        logger.info(f"Created stream: {name} ({stream_id})")
        
        return stream
    
    async def publish_data(
        self,
        stream_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Publicar datos en stream.
        
        Args:
            stream_id: ID del stream
            data: Datos a publicar
            
        Returns:
            True si se publicó, False si no existe
        """
        if stream_id not in self.streams:
            return False
        
        stream = self.streams[stream_id]
        
        if stream.status != StreamStatus.ACTIVE:
            return False
        
        # Publicar en cola principal
        await self.stream_queues[stream_id].put(data)
        
        # Publicar a todos los suscriptores
        for subscriber_queue in self.subscribers.get(stream_id, []):
            try:
                subscriber_queue.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning(f"Subscriber queue full for stream {stream_id}")
        
        stream.last_data_at = datetime.now().isoformat()
        
        return True
    
    async def subscribe(
        self,
        stream_id: str,
        subscriber_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Suscribirse a stream.
        
        Args:
            stream_id: ID del stream
            subscriber_id: ID del suscriptor
            
        Yields:
            Datos del stream
        """
        if stream_id not in self.streams:
            return
        
        stream = self.streams[stream_id]
        
        # Agregar suscriptor
        if subscriber_id not in stream.subscribers:
            stream.subscribers.append(subscriber_id)
        
        # Crear cola para suscriptor
        subscriber_queue = asyncio.Queue()
        if stream_id not in self.subscribers:
            self.subscribers[stream_id] = []
        self.subscribers[stream_id].append(subscriber_queue)
        
        try:
            while stream.status == StreamStatus.ACTIVE:
                try:
                    # Esperar datos con timeout
                    data = await asyncio.wait_for(
                        subscriber_queue.get(),
                        timeout=1.0
                    )
                    yield data
                except asyncio.TimeoutError:
                    # Enviar heartbeat
                    yield {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
        finally:
            # Remover suscriptor
            if subscriber_id in stream.subscribers:
                stream.subscribers.remove(subscriber_id)
            if subscriber_queue in self.subscribers.get(stream_id, []):
                self.subscribers[stream_id].remove(subscriber_queue)
    
    def get_stream(self, stream_id: str) -> Optional[Stream]:
        """Obtener stream por ID."""
        return self.streams.get(stream_id)
    
    def list_streams(
        self,
        status: Optional[StreamStatus] = None
    ) -> List[Stream]:
        """
        Listar streams.
        
        Args:
            status: Filtrar por estado
            
        Returns:
            Lista de streams
        """
        streams = list(self.streams.values())
        
        if status:
            streams = [s for s in streams if s.status == status]
        
        return streams
    
    def update_stream_status(
        self,
        stream_id: str,
        status: StreamStatus
    ) -> bool:
        """Actualizar estado de stream."""
        if stream_id not in self.streams:
            return False
        
        self.streams[stream_id].status = status
        return True
    
    def get_stream_statistics(self, stream_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de stream."""
        if stream_id not in self.streams:
            return {"error": "Stream not found"}
        
        stream = self.streams[stream_id]
        queue = self.stream_queues.get(stream_id)
        
        return {
            "stream_id": stream_id,
            "name": stream.name,
            "status": stream.status.value,
            "subscribers_count": len(stream.subscribers),
            "queue_size": queue.qsize() if queue else 0,
            "last_data_at": stream.last_data_at,
            "created_at": stream.created_at
        }


# Instancia global
_streaming_system: Optional[StreamingSystem] = None


def get_streaming_system() -> StreamingSystem:
    """Obtener instancia global del sistema de streaming."""
    global _streaming_system
    if _streaming_system is None:
        _streaming_system = StreamingSystem()
    return _streaming_system







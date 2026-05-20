"""
Real-time Service
=================

Servicio para notificaciones en tiempo real.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..api.websocket import WebSocketManager

logger = logging.getLogger(__name__)


class RealTimeService:
    """Servicio de tiempo real."""
    
    def __init__(self, ws_manager: Optional[WebSocketManager] = None):
        """
        Inicializar servicio.
        
        Args:
            ws_manager: Gestor WebSocket
        """
        self.ws_manager = ws_manager or WebSocketManager()
        self._logger = logger
    
    async def notify_event_created(self, artist_id: str, event: Dict[str, Any]):
        """
        Notificar creación de evento.
        
        Args:
            artist_id: ID del artista
            event: Datos del evento
        """
        message = {
            "type": "event_created",
            "artist_id": artist_id,
            "event": event,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.send_to_user(message, artist_id)
        self._logger.info(f"Notified artist {artist_id} about event creation")
    
    async def notify_event_updated(self, artist_id: str, event: Dict[str, Any]):
        """
        Notificar actualización de evento.
        
        Args:
            artist_id: ID del artista
            event: Datos del evento
        """
        message = {
            "type": "event_updated",
            "artist_id": artist_id,
            "event": event,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.send_to_user(message, artist_id)
        self._logger.info(f"Notified artist {artist_id} about event update")
    
    async def notify_routine_reminder(self, artist_id: str, routine: Dict[str, Any]):
        """
        Notificar recordatorio de rutina.
        
        Args:
            artist_id: ID del artista
            routine: Datos de la rutina
        """
        message = {
            "type": "routine_reminder",
            "artist_id": artist_id,
            "routine": routine,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.send_to_user(message, artist_id)
        self._logger.info(f"Notified artist {artist_id} about routine reminder")
    
    async def notify_protocol_alert(self, artist_id: str, protocol: Dict[str, Any]):
        """
        Notificar alerta de protocolo.
        
        Args:
            artist_id: ID del artista
            protocol: Datos del protocolo
        """
        message = {
            "type": "protocol_alert",
            "artist_id": artist_id,
            "protocol": protocol,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.send_to_user(message, artist_id)
        self._logger.info(f"Notified artist {artist_id} about protocol alert")
    
    async def broadcast_system_message(self, message: str, exclude: Optional[set] = None):
        """
        Broadcast mensaje del sistema.
        
        Args:
            message: Mensaje
            exclude: IDs a excluir
        """
        broadcast_msg = {
            "type": "system_message",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.broadcast(broadcast_msg, exclude)
        self._logger.info(f"Broadcasted system message: {message}")





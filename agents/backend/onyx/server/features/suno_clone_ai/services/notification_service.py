"""
Servicio de notificaciones para el sistema
"""

import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime

try:
    from api.websocket_api import notify_song_status, notify_generation_progress
except ImportError:
    from .api.websocket_api import notify_song_status, notify_generation_progress

logger = logging.getLogger(__name__)


class NotificationService:
    """Servicio para enviar notificaciones"""
    
    @staticmethod
    async def notify_song_completed(user_id: str, song_id: str, audio_url: str = None):
        """Notifica que una canción ha sido completada"""
        try:
            message = f"Song {song_id} has been generated successfully"
            if audio_url:
                message += f". Download at: {audio_url}"
            
            await notify_song_status(
                user_id=user_id,
                song_id=song_id,
                status="completed",
                message=message
            )
            logger.info(f"Notified user {user_id} about completed song {song_id}")
        except Exception as e:
            logger.error(f"Error notifying song completion: {e}")
    
    @staticmethod
    async def notify_song_failed(user_id: str, song_id: str, error_message: str = None):
        """Notifica que la generación de una canción falló"""
        try:
            message = f"Song {song_id} generation failed"
            if error_message:
                message += f": {error_message}"
            
            await notify_song_status(
                user_id=user_id,
                song_id=song_id,
                status="failed",
                message=message
            )
            logger.info(f"Notified user {user_id} about failed song {song_id}")
        except Exception as e:
            logger.error(f"Error notifying song failure: {e}")
    
    @staticmethod
    async def notify_generation_started(user_id: str, song_id: str):
        """Notifica que la generación ha comenzado"""
        try:
            await notify_song_status(
                user_id=user_id,
                song_id=song_id,
                status="processing",
                message=f"Song {song_id} generation started"
            )
            await notify_generation_progress(
                user_id=user_id,
                song_id=song_id,
                progress=0.0,
                message="Starting generation..."
            )
        except Exception as e:
            logger.error(f"Error notifying generation start: {e}")
    
    @staticmethod
    async def notify_progress(user_id: str, song_id: str, progress: float, stage: str = None):
        """Notifica el progreso de generación"""
        try:
            message = f"Generating... {progress * 100:.1f}%"
            if stage:
                message += f" ({stage})"
            
            await notify_generation_progress(
                user_id=user_id,
                song_id=song_id,
                progress=progress,
                message=message
            )
        except Exception as e:
            logger.error(f"Error notifying progress: {e}")


# Instancia global
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Obtiene la instancia global del servicio de notificaciones"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


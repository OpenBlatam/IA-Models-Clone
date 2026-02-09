"""
Notifications - Sistema de Notificaciones
==========================================

Sistema completo de notificaciones con múltiples canales y prioridades.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Prioridades de notificación"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    """Canales de notificación"""
    CONSOLE = "console"
    LOG = "log"
    WEBSOCKET = "websocket"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class Notification:
    """Notificación"""
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channel: NotificationChannel = NotificationChannel.CONSOLE
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Verificar si la notificación está expirada"""
        if self.expires_at:
            return datetime.now() >= self.expires_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "channel": self.channel.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class NotificationManager:
    """
    Gestor de notificaciones.
    
    Maneja múltiples canales de notificación con prioridades.
    """
    
    def __init__(self):
        self.channels: Dict[NotificationChannel, List[Callable]] = {}
        self.notifications: List[Notification] = []
        self.max_notifications = 10000
        self.enabled_channels: set[NotificationChannel] = {
            NotificationChannel.CONSOLE,
            NotificationChannel.LOG
        }
    
    def register_channel(
        self,
        channel: NotificationChannel,
        handler: Callable[[Notification], None]
    ) -> None:
        """
        Registrar handler para un canal.
        
        Args:
            channel: Canal de notificación
            handler: Función handler (puede ser async o sync)
        """
        if channel not in self.channels:
            self.channels[channel] = []
        
        self.channels[channel].append(handler)
        logger.info(f"📢 Notification channel registered: {channel.value}")
    
    async def send(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channel: Optional[NotificationChannel] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> Notification:
        """
        Enviar notificación.
        
        Args:
            title: Título de la notificación
            message: Mensaje
            priority: Prioridad
            channel: Canal específico (None = todos los habilitados)
            metadata: Metadata adicional
            expires_at: Fecha de expiración
            
        Returns:
            Notificación creada
        """
        notification = Notification(
            title=title,
            message=message,
            priority=priority,
            channel=channel or NotificationChannel.CONSOLE,
            metadata=metadata or {},
            expires_at=expires_at
        )
        
        # Agregar a historial
        self.notifications.append(notification)
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Enviar a canales
        channels_to_use = [notification.channel] if channel else list(self.enabled_channels)
        
        for channel in channels_to_use:
            if channel in self.channels:
                for handler in self.channels[channel]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(notification)
                        else:
                            handler(notification)
                    except Exception as e:
                        logger.error(f"Error sending notification to {channel.value}: {e}")
        
        # Log según prioridad
        log_level = {
            NotificationPriority.LOW: logger.debug,
            NotificationPriority.NORMAL: logger.info,
            NotificationPriority.HIGH: logger.warning,
            NotificationPriority.URGENT: logger.error
        }.get(priority, logger.info)
        
        log_level(f"📢 Notification: {title} - {message}")
        
        return notification
    
    def enable_channel(self, channel: NotificationChannel) -> None:
        """Habilitar canal de notificación"""
        self.enabled_channels.add(channel)
        logger.debug(f"📢 Channel enabled: {channel.value}")
    
    def disable_channel(self, channel: NotificationChannel) -> None:
        """Deshabilitar canal de notificación"""
        self.enabled_channels.discard(channel)
        logger.debug(f"📢 Channel disabled: {channel.value}")
    
    def get_notifications(
        self,
        channel: Optional[NotificationChannel] = None,
        priority: Optional[NotificationPriority] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Notification]:
        """
        Obtener notificaciones con filtros.
        
        Args:
            channel: Filtrar por canal
            priority: Filtrar por prioridad
            since: Filtrar desde fecha
            limit: Límite de resultados
            
        Returns:
            Lista de notificaciones
        """
        filtered = [n for n in self.notifications if not n.is_expired()]
        
        if channel:
            filtered = [n for n in filtered if n.channel == channel]
        
        if priority:
            filtered = [n for n in filtered if n.priority == priority]
        
        if since:
            filtered = [n for n in filtered if n.timestamp >= since]
        
        # Ordenar por prioridad y timestamp
        priority_order = {
            NotificationPriority.URGENT: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.NORMAL: 2,
            NotificationPriority.LOW: 3
        }
        
        filtered.sort(
            key=lambda x: (priority_order.get(x.priority, 99), x.timestamp),
            reverse=True
        )
        
        return filtered[:limit]
    
    def clear_expired(self) -> int:
        """
        Limpiar notificaciones expiradas.
        
        Returns:
            Número de notificaciones eliminadas
        """
        before = len(self.notifications)
        self.notifications = [n for n in self.notifications if not n.is_expired()]
        removed = before - len(self.notifications)
        
        if removed > 0:
            logger.debug(f"🧹 Cleared {removed} expired notifications")
        
        return removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones"""
        total = len(self.notifications)
        
        if total == 0:
            return {
                "total": 0,
                "by_priority": {},
                "by_channel": {},
                "active_channels": [c.value for c in self.enabled_channels]
            }
        
        by_priority = {}
        by_channel = {}
        
        for notification in self.notifications:
            priority = notification.priority.value
            channel = notification.channel.value
            
            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_channel[channel] = by_channel.get(channel, 0) + 1
        
        return {
            "total": total,
            "by_priority": by_priority,
            "by_channel": by_channel,
            "active_channels": [c.value for c in self.enabled_channels],
            "registered_channels": [c.value for c in self.channels.keys()]
        }

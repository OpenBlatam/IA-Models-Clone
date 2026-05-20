"""
Servicio de notificaciones para alertas y mensajes.
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime
import asyncio

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class NotificationLevel(str, Enum):
    """Niveles de notificación."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"


class NotificationChannel(str, Enum):
    """Canales de notificación."""
    LOG = "log"
    WEBSOCKET = "websocket"
    EMAIL = "email"
    WEBHOOK = "webhook"


class Notification:
    """
    Representa una notificación con validaciones.
    
    Attributes:
        title: Título de la notificación
        message: Mensaje de la notificación
        level: Nivel de severidad
        channels: Canales a usar
        metadata: Metadata adicional
        timestamp: Timestamp de creación
        id: ID único de la notificación
    """
    
    def __init__(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar notificación con validaciones.
        
        Args:
            title: Título de la notificación (debe ser string no vacío)
            message: Mensaje de la notificación (debe ser string no vacío)
            level: Nivel de severidad (default: INFO)
            channels: Canales a usar (default: LOG)
            metadata: Metadata adicional
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not title or not isinstance(title, str) or not title.strip():
            raise ValueError("title debe ser un string no vacío")
        
        if not message or not isinstance(message, str) or not message.strip():
            raise ValueError("message debe ser un string no vacío")
        
        if not isinstance(level, NotificationLevel):
            raise ValueError(f"level debe ser un NotificationLevel, recibido: {type(level)}")
        
        if channels is not None:
            if not isinstance(channels, list):
                raise ValueError("channels debe ser una lista si se proporciona")
            if len(channels) == 0:
                raise ValueError("channels no puede estar vacío si se proporciona")
            for channel in channels:
                if not isinstance(channel, NotificationChannel):
                    raise ValueError(f"Todos los canales deben ser NotificationChannel, recibido: {type(channel)}")
        
        self.title = title.strip()
        self.message = message.strip()
        self.level = level
        self.channels = channels or [NotificationChannel.LOG]
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.id = f"{self.timestamp}_{hash(self.title + self.message) % 10000}"
        
        logger.debug(f"Notificación creada: {self.title} (level: {self.level.value}, canales: {len(self.channels)})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir notificación a diccionario."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "channels": [c.value for c in self.channels],
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class NotificationService:
    """
    Servicio para enviar notificaciones a múltiples canales con mejoras.
    
    Attributes:
        notifications: Lista de notificaciones enviadas
        max_notifications: Número máximo de notificaciones a mantener
        handlers: Diccionario de handlers por canal
    """
    
    def __init__(self, max_notifications: int = 1000):
        """
        Inicializar servicio de notificaciones con validaciones.
        
        Args:
            max_notifications: Número máximo de notificaciones a mantener (default: 1000)
            
        Raises:
            ValueError: Si max_notifications es inválido
        """
        # Validación
        if not isinstance(max_notifications, int) or max_notifications < 1:
            raise ValueError(f"max_notifications debe ser un entero positivo, recibido: {max_notifications}")
        
        self.notifications: List[Notification] = []
        self.max_notifications = max_notifications
        self.handlers: Dict[NotificationChannel, List[Callable]] = {
            NotificationChannel.LOG: [self._log_notification],
            NotificationChannel.WEBSOCKET: [],
            NotificationChannel.EMAIL: [],
            NotificationChannel.WEBHOOK: []
        }
        
        logger.info(f"NotificationService inicializado (max_notifications: {max_notifications})")
    
    def register_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Notification], None]
    ) -> None:
        """
        Registrar handler para un canal con validaciones.
        
        Args:
            channel: Canal de notificación (debe ser NotificationChannel)
            handler: Función que maneja la notificación (debe ser callable)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(channel, NotificationChannel):
            raise ValueError(f"channel debe ser un NotificationChannel, recibido: {type(channel)}")
        
        if not callable(handler):
            raise ValueError(f"handler debe ser callable, recibido: {type(handler)}")
        
        if channel not in self.handlers:
            self.handlers[channel] = []
        
        self.handlers[channel].append(handler)
        logger.info(f"✅ Handler registrado para canal {channel.value} (total: {len(self.handlers[channel])})")
    
    async def send(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Enviar notificación con validaciones y mejor manejo de errores.
        
        Args:
            title: Título de la notificación (debe ser string no vacío)
            message: Mensaje (debe ser string no vacío)
            level: Nivel de severidad (default: INFO)
            channels: Canales a usar (default: LOG)
            metadata: Metadata adicional
            
        Returns:
            Notificación creada
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones básicas (Notification.__init__ también valida)
        if not title or not isinstance(title, str) or not title.strip():
            raise ValueError("title debe ser un string no vacío")
        
        if not message or not isinstance(message, str) or not message.strip():
            raise ValueError("message debe ser un string no vacío")
        
        try:
            notification = Notification(
                title=title,
                message=message,
                level=level,
                channels=channels,
                metadata=metadata
            )
        except ValueError as e:
            logger.error(f"Error al crear notificación: {e}", exc_info=True)
            raise
        
        # Agregar a historial
        self.notifications.append(notification)
        if len(self.notifications) > self.max_notifications:
            removed = self.notifications.pop(0)
            logger.debug(f"Notificación antigua removida del historial: {removed.id}")
        
        logger.debug(
            f"Enviando notificación '{notification.title}' "
            f"(level: {notification.level.value}, canales: {len(notification.channels)})"
        )
        
        # Enviar a cada canal
        successful_channels = []
        failed_channels = []
        
        for channel in notification.channels:
            handlers = self.handlers.get(channel, [])
            if not handlers:
                logger.warning(f"No hay handlers registrados para canal {channel.value}")
                failed_channels.append(channel.value)
                continue
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(notification)
                    else:
                        handler(notification)
                    successful_channels.append(channel.value)
                    logger.debug(f"✅ Notificación enviada exitosamente a canal {channel.value}")
                except Exception as e:
                    logger.error(
                        f"❌ Error en handler de notificación ({channel.value}): {e}",
                        exc_info=True
                    )
                    failed_channels.append(channel.value)
        
        if successful_channels:
            logger.info(
                f"✅ Notificación '{notification.title}' enviada a {len(set(successful_channels))} canales: "
                f"{', '.join(set(successful_channels))}"
            )
        
        if failed_channels:
            logger.warning(
                f"⚠️  Notificación '{notification.title}' falló en {len(set(failed_channels))} canales: "
                f"{', '.join(set(failed_channels))}"
            )
        
        return notification
    
    def _log_notification(self, notification: Notification) -> None:
        """Handler por defecto para logging."""
        log_level = {
            NotificationLevel.INFO: logger.info,
            NotificationLevel.SUCCESS: logger.info,
            NotificationLevel.WARNING: logger.warning,
            NotificationLevel.ERROR: logger.error,
            NotificationLevel.CRITICAL: logger.critical
        }.get(notification.level, logger.info)
        
        log_level(
            f"Notification [{notification.level.value}]: {notification.title} - {notification.message}",
            extra=notification.metadata
        )
    
    async def send_to_websocket(self, notification: Notification) -> None:
        """Enviar notificación vía WebSocket."""
        try:
            from api.routes.websocket_routes import manager
            await manager.broadcast_to_all({
                "type": "notification",
                "data": notification.to_dict()
            })
        except Exception as e:
            logger.debug(f"WebSocket notification failed: {e}")
    
    def get_notifications(
        self,
        level: Optional[NotificationLevel] = None,
        limit: int = 100
    ) -> List[Notification]:
        """
        Obtener notificaciones con validaciones.
        
        Args:
            level: Filtrar por nivel (opcional, debe ser NotificationLevel)
            limit: Número máximo de notificaciones (debe ser entero positivo, default: 100)
            
        Returns:
            Lista de notificaciones
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if level is not None and not isinstance(level, NotificationLevel):
            raise ValueError(f"level debe ser un NotificationLevel, recibido: {type(level)}")
        
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit debe ser un entero positivo, recibido: {limit}")
        
        # Limitar a máximo razonable
        if limit > 10000:
            logger.warning(f"limit muy alto ({limit}), limitando a 10000")
            limit = 10000
        
        filtered = self.notifications
        
        if level:
            filtered = [n for n in filtered if n.level == level]
        
        # Ordenar por timestamp (más recientes primero)
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        result = filtered[:limit]
        logger.debug(f"Obtenidas {len(result)} notificaciones (filtro: level={level}, limit={limit})")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones."""
        stats = {
            "total": len(self.notifications),
            "by_level": {},
            "by_channel": {},
            "latest": self.notifications[-1].to_dict() if self.notifications else None
        }
        
        for notification in self.notifications:
            # Por nivel
            level = notification.level.value
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # Por canal
            for channel in notification.channels:
                channel_name = channel.value
                stats["by_channel"][channel_name] = stats["by_channel"].get(channel_name, 0) + 1
        
        return stats


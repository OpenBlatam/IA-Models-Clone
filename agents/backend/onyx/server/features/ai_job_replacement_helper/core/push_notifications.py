"""
Push Notifications Service - Notificaciones push
=================================================

Sistema de notificaciones push para dispositivos móviles y web.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PushPlatform(str, Enum):
    """Plataformas de push"""
    WEB = "web"
    IOS = "ios"
    ANDROID = "android"


class PushPriority(str, Enum):
    """Prioridad de push"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class PushDevice:
    """Dispositivo para push"""
    device_id: str
    user_id: str
    platform: PushPlatform
    token: str
    enabled: bool = True
    registered_at: datetime = field(default_factory=datetime.now)


@dataclass
class PushNotification:
    """Notificación push"""
    id: str
    user_id: str
    title: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: PushPriority = PushPriority.NORMAL
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered: bool = False
    clicked: bool = False


class PushNotificationsService:
    """Servicio de notificaciones push"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.devices: Dict[str, List[PushDevice]] = {}  # user_id -> devices
        self.notifications: Dict[str, PushNotification] = {}
        logger.info("PushNotificationsService initialized")
    
    def register_device(
        self,
        user_id: str,
        device_id: str,
        platform: PushPlatform,
        token: str
    ) -> PushDevice:
        """Registrar dispositivo para push"""
        device = PushDevice(
            device_id=device_id,
            user_id=user_id,
            platform=platform,
            token=token,
        )
        
        if user_id not in self.devices:
            self.devices[user_id] = []
        
        # Actualizar si ya existe
        existing = next((d for d in self.devices[user_id] if d.device_id == device_id), None)
        if existing:
            existing.token = token
            existing.enabled = True
            return existing
        
        self.devices[user_id].append(device)
        
        logger.info(f"Device registered: {device_id} for user {user_id}")
        return device
    
    def send_push(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: PushPriority = PushPriority.NORMAL,
        scheduled_at: Optional[datetime] = None
    ) -> PushNotification:
        """Enviar notificación push"""
        notification_id = f"push_{user_id}_{int(datetime.now().timestamp())}"
        
        notification = PushNotification(
            id=notification_id,
            user_id=user_id,
            title=title,
            body=body,
            data=data or {},
            priority=priority,
            scheduled_at=scheduled_at,
        )
        
        # Si está programada, no enviar ahora
        if scheduled_at and scheduled_at > datetime.now():
            self.notifications[notification_id] = notification
            return notification
        
        # Enviar inmediatamente
        devices = self.devices.get(user_id, [])
        enabled_devices = [d for d in devices if d.enabled]
        
        if enabled_devices:
            # En producción, esto enviaría push real usando FCM, APNs, etc.
            notification.sent_at = datetime.now()
            notification.delivered = True
        
        self.notifications[notification_id] = notification
        
        logger.info(f"Push notification sent: {notification_id}")
        return notification
    
    def send_batch_push(
        self,
        user_ids: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ) -> List[PushNotification]:
        """Enviar push a múltiples usuarios"""
        notifications = []
        
        for user_id in user_ids:
            notification = self.send_push(user_id, title, body, data)
            notifications.append(notification)
        
        return notifications
    
    def mark_as_clicked(self, notification_id: str) -> bool:
        """Marcar notificación como clickeada"""
        notification = self.notifications.get(notification_id)
        if notification:
            notification.clicked = True
            return True
        return False
    
    def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtener notificaciones del usuario"""
        user_notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        
        # Ordenar por fecha (más recientes primero)
        user_notifications.sort(key=lambda n: n.sent_at or datetime.min, reverse=True)
        
        return [
            {
                "id": n.id,
                "title": n.title,
                "body": n.body,
                "sent_at": n.sent_at.isoformat() if n.sent_at else None,
                "delivered": n.delivered,
                "clicked": n.clicked,
            }
            for n in user_notifications[:limit]
        ]
    
    def unsubscribe_device(self, user_id: str, device_id: str) -> bool:
        """Desuscribir dispositivo"""
        devices = self.devices.get(user_id, [])
        device = next((d for d in devices if d.device_id == device_id), None)
        
        if device:
            device.enabled = False
            return True
        
        return False





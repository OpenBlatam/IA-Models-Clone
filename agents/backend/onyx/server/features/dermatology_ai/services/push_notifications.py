"""
Sistema de notificaciones push mejorado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class PushPlatform(str, Enum):
    """Plataformas de push"""
    FCM = "fcm"  # Firebase Cloud Messaging
    APNS = "apns"  # Apple Push Notification Service
    WEB_PUSH = "web_push"
    EMAIL = "email"
    SMS = "sms"


@dataclass
class PushNotification:
    """Notificación push"""
    id: str
    user_id: str
    platform: PushPlatform
    title: str
    body: str
    data: Optional[Dict] = None
    priority: str = "normal"  # "high", "normal", "low"
    sent: bool = False
    sent_at: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "platform": self.platform.value,
            "title": self.title,
            "body": self.body,
            "data": self.data or {},
            "priority": self.priority,
            "sent": self.sent,
            "sent_at": self.sent_at,
            "created_at": self.created_at
        }


class PushNotificationService:
    """Servicio de notificaciones push"""
    
    def __init__(self):
        """Inicializa el servicio"""
        self.notifications: Dict[str, PushNotification] = {}
        self.user_tokens: Dict[str, Dict[str, str]] = {}  # user_id -> {platform: token}
    
    def register_token(self, user_id: str, platform: PushPlatform, token: str):
        """
        Registra token de push
        
        Args:
            user_id: ID del usuario
            platform: Plataforma
            token: Token de push
        """
        if user_id not in self.user_tokens:
            self.user_tokens[user_id] = {}
        
        self.user_tokens[user_id][platform.value] = token
    
    def send_push(self, user_id: str, title: str, body: str,
                 platform: Optional[PushPlatform] = None,
                 data: Optional[Dict] = None,
                 priority: str = "normal") -> List[str]:
        """
        Envía notificación push
        
        Args:
            user_id: ID del usuario
            title: Título
            body: Cuerpo
            platform: Plataforma específica (opcional, envía a todas si None)
            data: Datos adicionales
            priority: Prioridad
            
        Returns:
            Lista de IDs de notificaciones enviadas
        """
        notification_ids = []
        
        # Obtener tokens del usuario
        user_tokens = self.user_tokens.get(user_id, {})
        
        platforms_to_send = [platform] if platform else [PushPlatform(p) for p in user_tokens.keys()]
        
        for platform in platforms_to_send:
            if platform.value not in user_tokens:
                continue
            
            notification_id = str(uuid.uuid4())
            
            notification = PushNotification(
                id=notification_id,
                user_id=user_id,
                platform=platform,
                title=title,
                body=body,
                data=data,
                priority=priority
            )
            
            # Enviar (placeholder - implementar con FCM, APNS, etc.)
            self._send_to_platform(platform, user_tokens[platform.value], notification)
            
            notification.sent = True
            notification.sent_at = datetime.now().isoformat()
            
            self.notifications[notification_id] = notification
            notification_ids.append(notification_id)
        
        return notification_ids
    
    def _send_to_platform(self, platform: PushPlatform, token: str,
                          notification: PushNotification):
        """Envía a plataforma específica"""
        # Placeholder - implementar con librerías reales
        if platform == PushPlatform.FCM:
            # Implementar con firebase-admin
            pass
        elif platform == PushPlatform.APNS:
            # Implementar con PyAPNs2
            pass
        elif platform == PushPlatform.WEB_PUSH:
            # Implementar con pywebpush
            pass
    
    def get_notification_history(self, user_id: str, limit: int = 50) -> List[PushNotification]:
        """Obtiene historial de notificaciones"""
        notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        
        notifications.sort(key=lambda x: x.created_at, reverse=True)
        return notifications[:limit]







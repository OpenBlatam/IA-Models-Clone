"""
Servicio de notificaciones
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificaciones"""
    ANALYSIS_COMPLETE = "analysis_complete"
    COACHING_READY = "coaching_ready"
    RECOMMENDATION_AVAILABLE = "recommendation_available"
    FAVORITE_UPDATE = "favorite_update"
    PLAYLIST_UPDATE = "playlist_update"
    SYSTEM_ANNOUNCEMENT = "system_announcement"


class NotificationPriority(Enum):
    """Prioridades de notificaciones"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationService:
    """Servicio para gestionar notificaciones"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/notifications")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.notifications_file = self.storage_path / "notifications.json"
        self.logger = logger
        self._load_notifications()
    
    def _load_notifications(self) -> None:
        """Carga notificaciones desde archivo"""
        if self.notifications_file.exists():
            try:
                with open(self.notifications_file, "r", encoding="utf-8") as f:
                    self.notifications = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading notifications: {e}")
                self.notifications = {}
        else:
            self.notifications = {}
    
    def _save_notifications(self) -> None:
        """Guarda notificaciones en archivo"""
        try:
            with open(self.notifications_file, "w", encoding="utf-8") as f:
                json.dump(self.notifications, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving notifications: {e}")
    
    def create_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Crea una nueva notificación"""
        notification_id = f"notif_{datetime.now().timestamp()}_{len(self.notifications)}"
        
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        notification = {
            "id": notification_id,
            "user_id": user_id,
            "type": notification_type.value,
            "title": title,
            "message": message,
            "priority": priority.value,
            "data": data or {},
            "read": False,
            "created_at": datetime.now().isoformat()
        }
        
        self.notifications[user_id].append(notification)
        self._save_notifications()
        
        self.logger.info(f"Notification created: {notification_id} for user {user_id}")
        return notification_id
    
    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtiene notificaciones de un usuario"""
        user_notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            user_notifications = [n for n in user_notifications if not n.get("read", False)]
        
        # Ordenar por fecha (más recientes primero)
        user_notifications.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return user_notifications[:limit]
    
    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Marca una notificación como leída"""
        user_notifications = self.notifications.get(user_id, [])
        
        for notification in user_notifications:
            if notification.get("id") == notification_id:
                notification["read"] = True
                self._save_notifications()
                return True
        
        return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Marca todas las notificaciones como leídas"""
        user_notifications = self.notifications.get(user_id, [])
        
        count = 0
        for notification in user_notifications:
            if not notification.get("read", False):
                notification["read"] = True
                count += 1
        
        if count > 0:
            self._save_notifications()
        
        return count
    
    def delete_notification(self, user_id: str, notification_id: str) -> bool:
        """Elimina una notificación"""
        user_notifications = self.notifications.get(user_id, [])
        
        original_count = len(user_notifications)
        self.notifications[user_id] = [
            n for n in user_notifications if n.get("id") != notification_id
        ]
        
        if len(self.notifications[user_id]) < original_count:
            self._save_notifications()
            return True
        
        return False
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtiene el número de notificaciones no leídas"""
        user_notifications = self.notifications.get(user_id, [])
        return sum(1 for n in user_notifications if not n.get("read", False))
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de notificaciones"""
        user_notifications = self.notifications.get(user_id, [])
        
        total = len(user_notifications)
        unread = sum(1 for n in user_notifications if not n.get("read", False))
        
        # Contar por tipo
        type_counts = {}
        for notification in user_notifications:
            notif_type = notification.get("type", "unknown")
            type_counts[notif_type] = type_counts.get(notif_type, 0) + 1
        
        # Contar por prioridad
        priority_counts = {}
        for notification in user_notifications:
            priority = notification.get("priority", "medium")
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            "total": total,
            "unread": unread,
            "read": total - unread,
            "by_type": type_counts,
            "by_priority": priority_counts
        }


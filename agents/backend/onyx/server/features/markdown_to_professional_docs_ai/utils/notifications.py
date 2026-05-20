"""Notification system"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manage notifications and alerts"""
    
    def __init__(self, notifications_dir: Optional[str] = None):
        """
        Initialize notification manager
        
        Args:
            notifications_dir: Directory for storing notifications
        """
        if notifications_dir is None:
            from config import settings
            notifications_dir = settings.temp_dir + "/notifications"
        
        self.notifications_dir = Path(notifications_dir)
        self.notifications_dir.mkdir(parents=True, exist_ok=True)
    
    def send_notification(
        self,
        notification_type: str,
        message: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ) -> str:
        """
        Send a notification
        
        Args:
            notification_type: Type (info, warning, error, success)
            message: Notification message
            recipient: Recipient identifier
            metadata: Additional metadata
            priority: Priority (low, normal, high, urgent)
            
        Returns:
            Notification ID
        """
        import uuid
        notification_id = str(uuid.uuid4())
        
        notification = {
            "id": notification_id,
            "type": notification_type,
            "message": message,
            "recipient": recipient,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "read": False,
            "metadata": metadata or {}
        }
        
        # Save notification
        notification_file = self.notifications_dir / f"{notification_id}.json"
        with open(notification_file, 'w') as f:
            json.dump(notification, f, indent=2)
        
        # Update recipient index
        if recipient:
            self._update_recipient_index(recipient, notification_id, notification)
        
        return notification_id
    
    def get_notifications(
        self,
        recipient: Optional[str] = None,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get notifications
        
        Args:
            recipient: Optional recipient filter
            unread_only: Only return unread notifications
            
        Returns:
            List of notifications
        """
        notifications = []
        
        if recipient:
            # Load from recipient index
            index_file = self.notifications_dir / f"{recipient}_index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
                    notifications = list(index.values())
        else:
            # Load all notifications
            for notification_file in self.notifications_dir.glob("*.json"):
                if notification_file.name.endswith("_index.json"):
                    continue
                
                with open(notification_file, 'r') as f:
                    notifications.append(json.load(f))
        
        # Filter unread
        if unread_only:
            notifications = [n for n in notifications if not n.get("read", False)]
        
        # Sort by creation date
        notifications.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return notifications
    
    def mark_as_read(
        self,
        notification_id: str
    ) -> bool:
        """
        Mark notification as read
        
        Args:
            notification_id: Notification ID
            
        Returns:
            True if marked, False otherwise
        """
        notification_file = self.notifications_dir / f"{notification_id}.json"
        
        if notification_file.exists():
            with open(notification_file, 'r') as f:
                notification = json.load(f)
            
            notification["read"] = True
            notification["read_at"] = datetime.now().isoformat()
            
            with open(notification_file, 'w') as f:
                json.dump(notification, f, indent=2)
            
            # Update recipient index
            recipient = notification.get("recipient")
            if recipient:
                self._update_recipient_index(recipient, notification_id, notification)
            
            return True
        
        return False
    
    def get_unread_count(
        self,
        recipient: Optional[str] = None
    ) -> int:
        """
        Get count of unread notifications
        
        Args:
            recipient: Optional recipient filter
            
        Returns:
            Count of unread notifications
        """
        notifications = self.get_notifications(recipient, unread_only=True)
        return len(notifications)
    
    def _update_recipient_index(
        self,
        recipient: str,
        notification_id: str,
        notification: Dict[str, Any]
    ) -> None:
        """Update recipient notification index"""
        index_file = self.notifications_dir / f"{recipient}_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {}
        
        index[notification_id] = notification
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)


# Global notification manager
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get global notification manager"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


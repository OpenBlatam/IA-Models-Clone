"""
Notification Mixin

Contains notification and alert functionality.
"""

import logging
from typing import Union, Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class NotificationMixin:
    """
    Mixin providing notification and alert functionality.
    
    This mixin contains:
    - Operation notifications
    - Progress notifications
    - Error notifications
    - Custom notification handlers
    - Notification history
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize notification mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_notification_handlers'):
            self._notification_handlers = []
        if not hasattr(self, '_notifications'):
            self._notifications = []
    
    def register_notification_handler(
        self,
        handler: Callable[[str, str, Dict[str, Any]], None]
    ) -> bool:
        """
        Register a notification handler.
        
        Args:
            handler: Function that takes (level, message, metadata)
            
        Returns:
            True if successful
        """
        if not callable(handler):
            return False
        
        self._notification_handlers.append(handler)
        logger.info("Notification handler registered")
        return True
    
    def notify(
        self,
        level: Union[str, NotificationLevel],
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a notification.
        
        Args:
            level: Notification level
            message: Notification message
            metadata: Optional metadata
        """
        if isinstance(level, str):
            try:
                level = NotificationLevel(level.lower())
            except ValueError:
                level = NotificationLevel.INFO
        
        notification = {
            "level": level.value,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Store notification
        self._notifications.append(notification)
        
        # Call handlers
        for handler in self._notification_handlers:
            try:
                handler(level.value, message, metadata or {})
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
        
        # Log based on level
        if level == NotificationLevel.ERROR:
            logger.error(message)
        elif level == NotificationLevel.WARNING:
            logger.warning(message)
        elif level == NotificationLevel.SUCCESS:
            logger.info(f"SUCCESS: {message}")
        else:
            logger.info(message)
    
    def notify_info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Send info notification."""
        self.notify(NotificationLevel.INFO, message, metadata)
    
    def notify_warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Send warning notification."""
        self.notify(NotificationLevel.WARNING, message, metadata)
    
    def notify_error(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Send error notification."""
        self.notify(NotificationLevel.ERROR, message, metadata)
    
    def notify_success(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Send success notification."""
        self.notify(NotificationLevel.SUCCESS, message, metadata)
    
    def notify_progress(
        self,
        operation: str,
        current: int,
        total: int,
        message: Optional[str] = None
    ) -> None:
        """
        Send progress notification.
        
        Args:
            operation: Operation name
            current: Current progress
            total: Total items
            message: Optional message
        """
        progress = (current / total * 100) if total > 0 else 0
        msg = message or f"{operation}: {current}/{total} ({progress:.1f}%)"
        
        self.notify(
            NotificationLevel.INFO,
            msg,
            {
                "operation": operation,
                "current": current,
                "total": total,
                "progress": progress,
            }
        )
    
    def get_notifications(
        self,
        level: Optional[Union[str, NotificationLevel]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get notifications.
        
        Args:
            level: Filter by level
            limit: Maximum number of notifications
            
        Returns:
            List of notifications
        """
        notifications = self._notifications.copy()
        
        if level:
            if isinstance(level, str):
                level = NotificationLevel(level.lower())
            notifications = [n for n in notifications if n["level"] == level.value]
        
        # Sort by timestamp (newest first)
        notifications.sort(key=lambda x: x["timestamp"], reverse=True)
        
        if limit:
            notifications = notifications[:limit]
        
        return notifications
    
    def clear_notifications(self) -> int:
        """
        Clear all notifications.
        
        Returns:
            Number of notifications cleared
        """
        count = len(self._notifications)
        self._notifications = []
        logger.info(f"Notifications cleared: {count}")
        return count



"""
Notification system for KV cache.

This module provides event notifications for cache operations,
enabling reactive programming patterns.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class NotificationType(Enum):
    """Notification types."""
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_SET = "cache_set"
    CACHE_DELETE = "cache_delete"
    CACHE_EXPIRE = "cache_expire"
    CACHE_EVICT = "cache_evict"
    CACHE_FULL = "cache_full"
    CACHE_ERROR = "cache_error"


@dataclass
class Notification:
    """A cache notification."""
    notification_type: NotificationType
    key: str
    value: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationSubscriber:
    """A notification subscriber."""
    
    def __init__(self, callback: Callable[[Notification], None], filter_types: Optional[Set[NotificationType]] = None):
        self.callback = callback
        self.filter_types = filter_types or set(NotificationType)
        self.subscribed_at = time.time()
        
    def should_receive(self, notification: Notification) -> bool:
        """Check if subscriber should receive this notification."""
        return notification.notification_type in self.filter_types


class CacheNotificationSystem:
    """Notification system for cache events."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._subscribers: List[NotificationSubscriber] = []
        self._lock = threading.Lock()
        
    def subscribe(
        self,
        callback: Callable[[Notification], None],
        notification_types: Optional[Set[NotificationType]] = None
    ) -> NotificationSubscriber:
        """Subscribe to cache notifications."""
        subscriber = NotificationSubscriber(callback, notification_types)
        
        with self._lock:
            self._subscribers.append(subscriber)
            
        return subscriber
        
    def unsubscribe(self, subscriber: NotificationSubscriber) -> bool:
        """Unsubscribe from notifications."""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                return True
            return False
            
    def notify(self, notification: Notification) -> None:
        """Notify all subscribers."""
        with self._lock:
            subscribers = list(self._subscribers)
            
        for subscriber in subscribers:
            if subscriber.should_receive(notification):
                try:
                    subscriber.callback(notification)
                except Exception as e:
                    print(f"Error in notification callback: {e}")
                    
    def notify_hit(self, key: str, value: Any) -> None:
        """Notify cache hit."""
        notification = Notification(
            notification_type=NotificationType.CACHE_HIT,
            key=key,
            value=value
        )
        self.notify(notification)
        
    def notify_miss(self, key: str) -> None:
        """Notify cache miss."""
        notification = Notification(
            notification_type=NotificationType.CACHE_MISS,
            key=key
        )
        self.notify(notification)
        
    def notify_set(self, key: str, value: Any) -> None:
        """Notify cache set."""
        notification = Notification(
            notification_type=NotificationType.CACHE_SET,
            key=key,
            value=value
        )
        self.notify(notification)
        
    def notify_delete(self, key: str) -> None:
        """Notify cache delete."""
        notification = Notification(
            notification_type=NotificationType.CACHE_DELETE,
            key=key
        )
        self.notify(notification)
        
    def notify_expire(self, key: str) -> None:
        """Notify cache expiration."""
        notification = Notification(
            notification_type=NotificationType.CACHE_EXPIRE,
            key=key
        )
        self.notify(notification)
        
    def notify_evict(self, key: str) -> None:
        """Notify cache eviction."""
        notification = Notification(
            notification_type=NotificationType.CACHE_EVICT,
            key=key
        )
        self.notify(notification)
        
    def get_subscriber_count(self) -> int:
        """Get number of subscribers."""
        return len(self._subscribers)


class NotifiedCache:
    """Cache wrapper with notification support."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.notification_system = CacheNotificationSystem(cache)
        
    def get(self, key: str) -> Any:
        """Get value and notify."""
        value = self.cache.get(key)
        if value is not None:
            self.notification_system.notify_hit(key, value)
        else:
            self.notification_system.notify_miss(key)
        return value
        
    def put(self, key: str, value: Any) -> bool:
        """Put value and notify."""
        result = self.cache.put(key, value)
        if result:
            self.notification_system.notify_set(key, value)
        return result
        
    def delete(self, key: str) -> bool:
        """Delete value and notify."""
        result = self.cache.delete(key)
        if result:
            self.notification_system.notify_delete(key)
        return result
        
    def subscribe(
        self,
        callback: Callable[[Notification], None],
        notification_types: Optional[Set[NotificationType]] = None
    ) -> NotificationSubscriber:
        """Subscribe to notifications."""
        return self.notification_system.subscribe(callback, notification_types)




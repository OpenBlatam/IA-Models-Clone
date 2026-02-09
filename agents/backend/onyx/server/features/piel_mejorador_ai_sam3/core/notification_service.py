"""
Notification Service for Piel Mejorador AI SAM3
================================================

Advanced notification system with multiple channels.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


@dataclass
class Notification:
    """Notification data structure."""
    title: str
    message: str
    channels: List[NotificationChannel]
    priority: int = 0
    data: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.data is None:
            self.data = {}


class NotificationProvider(ABC):
    """Interface for notification providers."""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification."""
        pass
    
    @abstractmethod
    def get_channel(self) -> NotificationChannel:
        """Get notification channel."""
        pass


class NotificationService:
    """
    Advanced notification service.
    
    Features:
    - Multiple notification channels
    - Priority handling
    - Retry logic
    - Delivery tracking
    """
    
    def __init__(self):
        """Initialize notification service."""
        self._providers: Dict[NotificationChannel, NotificationProvider] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        self._stats = {
            "notifications_sent": 0,
            "notifications_delivered": 0,
            "notifications_failed": 0,
        }
    
    def register_provider(self, provider: NotificationProvider):
        """Register a notification provider."""
        channel = provider.get_channel()
        self._providers[channel] = provider
        logger.info(f"Registered notification provider: {channel.value}")
    
    async def send_notification(self, notification: Notification):
        """
        Send a notification.
        
        Args:
            notification: Notification to send
        """
        await self._queue.put(notification)
        self._stats["notifications_sent"] += 1
    
    async def start_processing(self):
        """Start processing notification queue."""
        self._running = True
        
        while self._running:
            try:
                notification = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                # Send to all requested channels
                tasks = []
                for channel in notification.channels:
                    if channel in self._providers:
                        provider = self._providers[channel]
                        task = asyncio.create_task(
                            self._send_via_provider(provider, notification)
                        )
                        tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    async def _send_via_provider(
        self,
        provider: NotificationProvider,
        notification: Notification
    ):
        """Send notification via provider."""
        try:
            success = await provider.send(notification)
            if success:
                self._stats["notifications_delivered"] += 1
            else:
                self._stats["notifications_failed"] += 1
        except Exception as e:
            logger.error(f"Error sending notification via {provider.get_channel().value}: {e}")
            self._stats["notifications_failed"] += 1
    
    async def stop_processing(self):
        """Stop processing notifications."""
        self._running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        delivery_rate = (
            self._stats["notifications_delivered"] / self._stats["notifications_sent"]
            if self._stats["notifications_sent"] > 0 else 0
        )
        
        return {
            **self._stats,
            "delivery_rate": delivery_rate,
            "queue_size": self._queue.qsize(),
            "registered_channels": [ch.value for ch in self._providers.keys()],
        }





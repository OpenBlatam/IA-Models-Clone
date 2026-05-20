"""
Tests for Push Notifications
============================
"""

import pytest
import asyncio
from ..core.push_notifications import PushNotificationManager, NotificationChannel, NotificationPriority


@pytest.fixture
def push_notification_manager():
    """Create push notification manager for testing."""
    return PushNotificationManager()


@pytest.mark.asyncio
async def test_send_notification(push_notification_manager):
    """Test sending a push notification."""
    notification_id = await push_notification_manager.send_notification(
        user_id="test_user",
        title="Test Notification",
        message="This is a test notification",
        channel=NotificationChannel.WEB,
        priority=NotificationPriority.MEDIUM
    )
    
    assert notification_id is not None
    assert notification_id in push_notification_manager.notifications


@pytest.mark.asyncio
async def test_send_to_multiple_channels(push_notification_manager):
    """Test sending to multiple channels."""
    notification_id = await push_notification_manager.send_notification(
        "test_user",
        "Test",
        "Message",
        channel=NotificationChannel.ALL,
        priority=NotificationPriority.HIGH
    )
    
    assert notification_id is not None


@pytest.mark.asyncio
async def test_get_user_notifications(push_notification_manager):
    """Test getting notifications for a user."""
    await push_notification_manager.send_notification("user1", "Title 1", "Message 1")
    await push_notification_manager.send_notification("user1", "Title 2", "Message 2")
    await push_notification_manager.send_notification("user2", "Title 3", "Message 3")
    
    notifications = push_notification_manager.get_user_notifications("user1", limit=10)
    
    assert len(notifications) >= 2
    assert all(n.user_id == "user1" for n in notifications)


@pytest.mark.asyncio
async def test_mark_notification_read(push_notification_manager):
    """Test marking notification as read."""
    notification_id = await push_notification_manager.send_notification(
        "test_user", "Test", "Message"
    )
    
    push_notification_manager.mark_as_read(notification_id)
    
    notification = push_notification_manager.get_notification(notification_id)
    assert notification.read is True


@pytest.mark.asyncio
async def test_subscribe_to_channel(push_notification_manager):
    """Test subscribing to a notification channel."""
    subscription_id = push_notification_manager.subscribe(
        user_id="test_user",
        channel=NotificationChannel.EMAIL
    )
    
    assert subscription_id is not None


@pytest.mark.asyncio
async def test_get_push_notification_summary(push_notification_manager):
    """Test getting push notification summary."""
    await push_notification_manager.send_notification("user1", "Title 1", "Message 1")
    await push_notification_manager.send_notification("user2", "Title 2", "Message 2")
    
    summary = push_notification_manager.get_push_notification_summary()
    
    assert summary is not None
    assert "total_notifications" in summary or "notifications_by_channel" in summary



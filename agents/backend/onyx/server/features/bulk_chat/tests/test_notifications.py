"""
Tests for Notification Manager
===============================
"""

import pytest
import asyncio
from ..core.notifications import NotificationManager, NotificationType, NotificationPriority


@pytest.fixture
def notification_manager():
    """Create notification manager for testing."""
    return NotificationManager()


@pytest.mark.asyncio
async def test_create_notification(notification_manager):
    """Test creating a notification."""
    notification_id = notification_manager.create_notification(
        user_id="test_user",
        title="Test Notification",
        message="This is a test",
        notification_type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM
    )
    
    assert notification_id is not None
    assert notification_id in notification_manager.notifications


@pytest.mark.asyncio
async def test_get_notifications(notification_manager):
    """Test getting notifications for a user."""
    notification_manager.create_notification("user1", "Title 1", "Message 1")
    notification_manager.create_notification("user1", "Title 2", "Message 2")
    notification_manager.create_notification("user2", "Title 3", "Message 3")
    
    notifications = notification_manager.get_notifications("user1", limit=10)
    
    assert len(notifications) >= 2
    assert all(n.user_id == "user1" for n in notifications)


@pytest.mark.asyncio
async def test_mark_notification_read(notification_manager):
    """Test marking notification as read."""
    notification_id = notification_manager.create_notification(
        "test_user",
        "Test",
        "Message"
    )
    
    notification_manager.mark_as_read(notification_id)
    
    notification = notification_manager.get_notification(notification_id)
    assert notification.read is True


@pytest.mark.asyncio
async def test_delete_notification(notification_manager):
    """Test deleting a notification."""
    notification_id = notification_manager.create_notification(
        "test_user",
        "Test",
        "Message"
    )
    
    assert notification_id in notification_manager.notifications
    
    notification_manager.delete_notification(notification_id)
    
    assert notification_id not in notification_manager.notifications


@pytest.mark.asyncio
async def test_get_unread_count(notification_manager):
    """Test getting unread notification count."""
    notification_manager.create_notification("user1", "Title 1", "Message 1")
    notification_manager.create_notification("user1", "Title 2", "Message 2")
    
    # Mark one as read
    notifications = notification_manager.get_notifications("user1")
    if notifications:
        notification_manager.mark_as_read(notifications[0].notification_id)
    
    unread_count = notification_manager.get_unread_count("user1")
    
    assert unread_count >= 1



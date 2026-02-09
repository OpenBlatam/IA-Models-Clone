"""
Notification Testing Helpers
Specialized helpers for notification testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import uuid


class NotificationTestHelpers:
    """Helpers for notification testing"""
    
    @staticmethod
    def create_mock_notification_service(
        notifications_sent: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock notification service"""
        notifications = notifications_sent or []
        service = Mock()
        
        async def send_side_effect(
            recipient: str,
            subject: str,
            message: str,
            notification_type: str = "email"
        ):
            notifications.append({
                "id": str(uuid.uuid4()),
                "recipient": recipient,
                "subject": subject,
                "message": message,
                "type": notification_type,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "sent"
            })
            return True
        
        service.send = AsyncMock(side_effect=send_side_effect)
        service.send_batch = AsyncMock(return_value=True)
        service.notifications = notifications
        return service
    
    @staticmethod
    def assert_notification_sent(
        service: Mock,
        recipient: str,
        subject: Optional[str] = None
    ):
        """Assert notification was sent"""
        assert service.send.called, "Notification was not sent"
        
        if hasattr(service, "notifications"):
            matching = [
                n for n in service.notifications
                if n["recipient"] == recipient
            ]
            assert len(matching) > 0, f"No notifications sent to {recipient}"
            
            if subject:
                found = any(n["subject"] == subject for n in matching)
                assert found, f"Notification with subject '{subject}' not found"


class EmailHelpers:
    """Helpers for email testing"""
    
    @staticmethod
    def create_mock_email_service() -> Mock:
        """Create mock email service"""
        service = Mock()
        service.send_email = AsyncMock(return_value=True)
        service.send_bulk_email = AsyncMock(return_value=True)
        service.validate_email = Mock(return_value=True)
        return service
    
    @staticmethod
    def assert_email_sent(
        service: Mock,
        to: str,
        subject: Optional[str] = None
    ):
        """Assert email was sent"""
        assert service.send_email.called, "Email was not sent"
        # Additional validation can check call arguments


class SMSHelpers:
    """Helpers for SMS testing"""
    
    @staticmethod
    def create_mock_sms_service() -> Mock:
        """Create mock SMS service"""
        service = Mock()
        service.send_sms = AsyncMock(return_value=True)
        service.send_bulk_sms = AsyncMock(return_value=True)
        service.validate_phone = Mock(return_value=True)
        return service
    
    @staticmethod
    def assert_sms_sent(
        service: Mock,
        to: str,
        message: Optional[str] = None
    ):
        """Assert SMS was sent"""
        assert service.send_sms.called, "SMS was not sent"


class PushNotificationHelpers:
    """Helpers for push notification testing"""
    
    @staticmethod
    def create_mock_push_service() -> Mock:
        """Create mock push notification service"""
        service = Mock()
        service.send_push = AsyncMock(return_value=True)
        service.send_to_device = AsyncMock(return_value=True)
        service.send_to_topic = AsyncMock(return_value=True)
        return service
    
    @staticmethod
    def assert_push_sent(
        service: Mock,
        device_id: Optional[str] = None,
        topic: Optional[str] = None
    ):
        """Assert push notification was sent"""
        if device_id:
            assert service.send_to_device.called, "Push to device was not sent"
        elif topic:
            assert service.send_to_topic.called, "Push to topic was not sent"
        else:
            assert service.send_push.called, "Push notification was not sent"


# Convenience exports
create_mock_notification_service = NotificationTestHelpers.create_mock_notification_service
assert_notification_sent = NotificationTestHelpers.assert_notification_sent

create_mock_email_service = EmailHelpers.create_mock_email_service
assert_email_sent = EmailHelpers.assert_email_sent

create_mock_sms_service = SMSHelpers.create_mock_sms_service
assert_sms_sent = SMSHelpers.assert_sms_sent

create_mock_push_service = PushNotificationHelpers.create_mock_push_service
assert_push_sent = PushNotificationHelpers.assert_push_sent




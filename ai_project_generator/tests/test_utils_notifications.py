"""
Tests for NotificationService utility
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from ..utils.notification_service import NotificationService


class TestNotificationService:
    """Test suite for NotificationService"""

    def test_init(self):
        """Test NotificationService initialization"""
        service = NotificationService()
        assert "email" in service.channels
        assert "slack" in service.channels
        assert "discord" in service.channels
        assert "telegram" in service.channels

    def test_register_channel_slack(self):
        """Test registering Slack channel"""
        service = NotificationService()
        
        channel_id = service.register_channel(
            "slack",
            {"webhook_url": "https://hooks.slack.com/test"}
        )
        
        assert channel_id is not None
        assert len(service.channels["slack"]) == 1
        assert service.channels["slack"][0]["config"]["webhook_url"] == "https://hooks.slack.com/test"

    def test_register_channel_discord(self):
        """Test registering Discord channel"""
        service = NotificationService()
        
        channel_id = service.register_channel(
            "discord",
            {"webhook_url": "https://discord.com/api/webhooks/test"}
        )
        
        assert channel_id is not None
        assert len(service.channels["discord"]) == 1

    def test_register_channel_telegram(self):
        """Test registering Telegram channel"""
        service = NotificationService()
        
        channel_id = service.register_channel(
            "telegram",
            {"bot_token": "test_token", "chat_id": "123456"}
        )
        
        assert channel_id is not None
        assert len(service.channels["telegram"]) == 1

    def test_register_channel_email(self):
        """Test registering email channel"""
        service = NotificationService()
        
        channel_id = service.register_channel(
            "email",
            {"smtp_server": "smtp.example.com", "from_email": "test@example.com"}
        )
        
        assert channel_id is not None
        assert len(service.channels["email"]) == 1

    def test_register_channel_invalid(self):
        """Test registering invalid channel type"""
        service = NotificationService()
        
        with pytest.raises(ValueError, match="no soportado"):
            service.register_channel("invalid", {})

    @pytest.mark.asyncio
    async def test_send_notification_slack(self):
        """Test sending notification to Slack"""
        service = NotificationService()
        
        service.register_channel("slack", {"webhook_url": "https://hooks.slack.com/test"})
        
        with patch.object(service, '_send_slack', new_callable=AsyncMock) as mock_send:
            await service.send_notification(
                "Test message",
                title="Test Title",
                channels=["slack"]
            )
            
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_all_channels(self):
        """Test sending notification to all channels"""
        service = NotificationService()
        
        service.register_channel("slack", {"webhook_url": "https://hooks.slack.com/test"})
        service.register_channel("discord", {"webhook_url": "https://discord.com/api/webhooks/test"})
        
        with patch.object(service, '_send_slack', new_callable=AsyncMock) as mock_slack, \
             patch.object(service, '_send_discord', new_callable=AsyncMock) as mock_discord:
            
            await service.send_notification("Test message")
            
            # Should send to all active channels
            assert mock_slack.called or mock_discord.called

    @pytest.mark.asyncio
    async def test_send_notification_priority(self):
        """Test sending notification with priority"""
        service = NotificationService()
        
        service.register_channel("slack", {"webhook_url": "https://hooks.slack.com/test"})
        
        with patch.object(service, '_send_slack', new_callable=AsyncMock):
            await service.send_notification(
                "Urgent message",
                priority="urgent"
            )
            # Should handle priority

    @pytest.mark.asyncio
    async def test_send_notification_inactive_channel(self):
        """Test sending to inactive channel"""
        service = NotificationService()
        
        channel_id = service.register_channel("slack", {"webhook_url": "https://hooks.slack.com/test"})
        service.channels["slack"][0]["active"] = False
        
        with patch.object(service, '_send_slack', new_callable=AsyncMock) as mock_send:
            await service.send_notification("Test", channels=["slack"])
            
            # Should not send to inactive channel
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_notification_error_handling(self):
        """Test error handling in notification sending"""
        service = NotificationService()
        
        service.register_channel("slack", {"webhook_url": "https://hooks.slack.com/test"})
        
        with patch.object(service, '_send_slack', new_callable=AsyncMock, side_effect=Exception("API Error")):
            # Should handle error gracefully
            await service.send_notification("Test", channels=["slack"])
            # Should not crash

    def test_list_channels(self):
        """Test listing channels"""
        service = NotificationService()
        
        service.register_channel("slack", {"webhook_url": "test1"})
        service.register_channel("slack", {"webhook_url": "test2"})
        service.register_channel("discord", {"webhook_url": "test3"})
        
        channels = service.list_channels()
        
        assert len(channels) >= 3


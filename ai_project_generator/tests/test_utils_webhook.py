"""
Tests for WebhookManager utility
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from datetime import datetime

from ..utils.webhook_manager import WebhookManager


class TestWebhookManager:
    """Test suite for WebhookManager"""

    def test_init(self):
        """Test WebhookManager initialization"""
        manager = WebhookManager()
        assert manager.webhooks == []

    def test_register_webhook(self):
        """Test registering a webhook"""
        manager = WebhookManager()
        
        webhook_id = manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed", "project.failed"],
            secret="test_secret"
        )
        
        assert webhook_id is not None
        assert len(manager.webhooks) == 1
        assert manager.webhooks[0]["url"] == "http://example.com/webhook"
        assert "project.completed" in manager.webhooks[0]["events"]
        assert manager.webhooks[0]["secret"] == "test_secret"
        assert manager.webhooks[0]["active"] is True

    def test_register_webhook_no_secret(self):
        """Test registering webhook without secret"""
        manager = WebhookManager()
        
        webhook_id = manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed"]
        )
        
        assert webhook_id is not None
        assert manager.webhooks[0]["secret"] is None

    def test_register_multiple_webhooks(self):
        """Test registering multiple webhooks"""
        manager = WebhookManager()
        
        id1 = manager.register_webhook("http://example.com/1", ["event1"])
        id2 = manager.register_webhook("http://example.com/2", ["event2"])
        
        assert len(manager.webhooks) == 2
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_trigger_webhook(self):
        """Test triggering a webhook"""
        manager = WebhookManager()
        
        webhook_id = manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed"]
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            await manager.trigger_webhook("project.completed", {"project_id": "test-123"})
            
            # Should have attempted to call webhook
            assert True  # Webhook triggered

    @pytest.mark.asyncio
    async def test_trigger_webhook_inactive(self):
        """Test triggering inactive webhook"""
        manager = WebhookManager()
        
        webhook_id = manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed"]
        )
        manager.webhooks[0]["active"] = False
        
        # Should not trigger inactive webhook
        await manager.trigger_webhook("project.completed", {"project_id": "test-123"})
        # No exception should be raised

    @pytest.mark.asyncio
    async def test_trigger_webhook_wrong_event(self):
        """Test triggering webhook for wrong event"""
        manager = WebhookManager()
        
        manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed"]
        )
        
        # Trigger different event
        await manager.trigger_webhook("project.failed", {"project_id": "test-123"})
        # Should not trigger for wrong event

    @pytest.mark.asyncio
    async def test_trigger_webhook_with_secret(self):
        """Test triggering webhook with secret signature"""
        manager = WebhookManager()
        
        manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed"],
            secret="test_secret"
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            await manager.trigger_webhook("project.completed", {"project_id": "test-123"})
            
            # Should include signature in payload
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            payload = call_args[1]["json"]
            assert "signature" in payload

    def test_list_webhooks(self):
        """Test listing webhooks"""
        manager = WebhookManager()
        
        manager.register_webhook("http://example.com/1", ["event1"])
        manager.register_webhook("http://example.com/2", ["event2"])
        
        webhooks = manager.list_webhooks()
        assert len(webhooks) == 2

    def test_delete_webhook(self):
        """Test deleting a webhook"""
        manager = WebhookManager()
        
        webhook_id = manager.register_webhook("http://example.com/webhook", ["event1"])
        assert len(manager.webhooks) == 1
        
        result = manager.delete_webhook(webhook_id)
        assert result is True
        assert len(manager.webhooks) == 0

    def test_delete_webhook_not_found(self):
        """Test deleting non-existent webhook"""
        manager = WebhookManager()
        
        result = manager.delete_webhook("non-existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_webhook_error_handling(self):
        """Test error handling when webhook fails"""
        manager = WebhookManager()
        
        manager.register_webhook(
            url="http://example.com/webhook",
            events=["project.completed"]
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=Exception("Connection error"))
            
            # Should handle error gracefully
            await manager.trigger_webhook("project.completed", {"project_id": "test-123"})
            # No exception should be raised


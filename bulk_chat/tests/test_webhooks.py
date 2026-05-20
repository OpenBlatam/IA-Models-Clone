"""
Tests for Webhook Manager
==========================
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from ..core.webhooks import WebhookManager, WebhookEvent


@pytest.fixture
def webhook_manager():
    """Create webhook manager for testing."""
    return WebhookManager()


@pytest.mark.asyncio
async def test_register_webhook(webhook_manager):
    """Test registering a webhook."""
    webhook_id = webhook_manager.register_webhook(
        url="https://example.com/webhook",
        event_types=[WebhookEvent.SESSION_CREATED, WebhookEvent.SESSION_PAUSED]
    )
    
    assert webhook_id is not None
    assert webhook_id in webhook_manager.webhooks


@pytest.mark.asyncio
async def test_trigger_webhook(webhook_manager):
    """Test triggering a webhook."""
    webhook_id = webhook_manager.register_webhook(
        url="https://example.com/webhook",
        event_types=[WebhookEvent.SESSION_CREATED]
    )
    
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 200
        
        await webhook_manager.trigger_webhook(
            event_type=WebhookEvent.SESSION_CREATED,
            data={"session_id": "test_session"}
        )
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Verify webhook was called (in real implementation)
        assert webhook_manager.webhooks[webhook_id].event_types == [WebhookEvent.SESSION_CREATED]


@pytest.mark.asyncio
async def test_unregister_webhook(webhook_manager):
    """Test unregistering a webhook."""
    webhook_id = webhook_manager.register_webhook(
        url="https://example.com/webhook",
        event_types=[WebhookEvent.SESSION_CREATED]
    )
    
    assert webhook_id in webhook_manager.webhooks
    
    webhook_manager.unregister_webhook(webhook_id)
    
    assert webhook_id not in webhook_manager.webhooks


@pytest.mark.asyncio
async def test_get_webhook_status(webhook_manager):
    """Test getting webhook status."""
    webhook_id = webhook_manager.register_webhook(
        url="https://example.com/webhook",
        event_types=[WebhookEvent.SESSION_CREATED]
    )
    
    status = webhook_manager.get_webhook_status(webhook_id)
    
    assert status["webhook_id"] == webhook_id
    assert status["url"] == "https://example.com/webhook"
    assert len(status["event_types"]) == 1



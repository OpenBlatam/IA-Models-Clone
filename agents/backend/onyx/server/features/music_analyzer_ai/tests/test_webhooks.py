"""
Tests para WebhookService
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime


class TestWebhookService:
    """Tests para WebhookService"""
    
    @pytest.fixture
    def webhook_service(self):
        """Fixture para crear WebhookService"""
        from ..services.webhook_service import WebhookService, WebhookEvent
        return WebhookService()
    
    def test_register_webhook(self, webhook_service):
        """Test de registro de webhook"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED],
            secret="secret123",
            user_id="user123"
        )
        
        assert webhook_id is not None
        assert webhook_id in webhook_service.webhooks
        assert webhook_service.webhooks[webhook_id]["url"] == "https://example.com/webhook"
        assert webhook_service.webhooks[webhook_id]["active"] == True
    
    def test_get_webhook(self, webhook_service):
        """Test de obtención de webhook"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED]
        )
        
        webhook = webhook_service.get_webhook(webhook_id)
        
        assert webhook is not None
        assert webhook["id"] == webhook_id
    
    def test_get_webhook_not_found(self, webhook_service):
        """Test de obtención de webhook inexistente"""
        webhook = webhook_service.get_webhook("nonexistent_id")
        
        assert webhook is None
    
    def test_list_webhooks(self, webhook_service):
        """Test de listado de webhooks"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_service.register_webhook(
            url="https://example.com/webhook1",
            events=[WebhookEvent.ANALYSIS_COMPLETED],
            user_id="user123"
        )
        webhook_service.register_webhook(
            url="https://example.com/webhook2",
            events=[WebhookEvent.COMPARISON_COMPLETED],
            user_id="user123"
        )
        
        webhooks = webhook_service.list_webhooks(user_id="user123")
        
        assert len(webhooks) == 2
    
    def test_delete_webhook(self, webhook_service):
        """Test de eliminación de webhook"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED]
        )
        
        success = webhook_service.delete_webhook(webhook_id)
        
        assert success == True
        assert webhook_id not in webhook_service.webhooks
    
    def test_delete_webhook_not_found(self, webhook_service):
        """Test de eliminación de webhook inexistente"""
        success = webhook_service.delete_webhook("nonexistent_id")
        
        assert success == False
    
    @pytest.mark.asyncio
    async def test_trigger_webhook(self, webhook_service):
        """Test de activación de webhook"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED]
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"success": True})
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await webhook_service.trigger_webhook(
                webhook_id,
                WebhookEvent.ANALYSIS_COMPLETED,
                {"track_id": "123", "status": "completed"}
            )
            
            # Verificar que se actualizó el contador
            webhook = webhook_service.webhooks[webhook_id]
            assert webhook["success_count"] > 0
    
    def test_trigger_webhook_for_event(self, webhook_service):
        """Test de activación de webhooks para un evento"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED, WebhookEvent.COMPARISON_COMPLETED]
        )
        
        # Simular trigger (sin hacer request real)
        webhooks = webhook_service._get_webhooks_for_event(WebhookEvent.ANALYSIS_COMPLETED)
        
        assert len(webhooks) >= 1
        assert any(wh["id"] == webhook_id for wh in webhooks)
    
    def test_deactivate_webhook(self, webhook_service):
        """Test de desactivación de webhook"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED]
        )
        
        success = webhook_service.deactivate_webhook(webhook_id)
        
        assert success == True
        assert webhook_service.webhooks[webhook_id]["active"] == False
    
    def test_activate_webhook(self, webhook_service):
        """Test de activación de webhook"""
        from ..services.webhook_service import WebhookEvent
        
        webhook_id = webhook_service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED]
        )
        
        webhook_service.deactivate_webhook(webhook_id)
        success = webhook_service.activate_webhook(webhook_id)
        
        assert success == True
        assert webhook_service.webhooks[webhook_id]["active"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Tests para las rutas de webhooks
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.webhooks import router
from services.webhooks import WebhookService, WebhookEvent


@pytest.fixture
def mock_webhook_service():
    """Mock del servicio de webhooks"""
    service = Mock(spec=WebhookService)
    service.register_webhook = Mock(return_value="webhook-123")
    service.list_webhooks = Mock(return_value=[
        {
            "webhook_id": "webhook-1",
            "url": "https://example.com/webhook",
            "events": ["song.completed"]
        }
    ])
    service.get_stats = Mock(return_value={
        "total_webhooks": 5,
        "total_calls": 100
    })
    service.delete_webhook = Mock(return_value=True)
    return service


@pytest.fixture
def client(mock_webhook_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.webhooks.get_webhook_service', return_value=mock_webhook_service):
        with patch('api.routes.webhooks.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestRegisterWebhook:
    """Tests para registrar webhook"""
    
    def test_register_webhook_success(self, client, mock_webhook_service):
        """Test de registro exitoso de webhook"""
        response = client.post(
            "/webhooks/register",
            json={
                "url": "https://example.com/webhook",
                "events": ["song.completed", "song.failed"],
                "secret": "secret-key"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "webhook_id" in data
        assert data["message"] == "Webhook registered successfully"
        assert data["url"] == "https://example.com/webhook"
        assert len(data["events"]) == 2
    
    def test_register_webhook_invalid_event(self, client):
        """Test con evento inválido"""
        response = client.post(
            "/webhooks/register",
            json={
                "url": "https://example.com/webhook",
                "events": ["invalid.event"]
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid event type" in response.json()["detail"]
    
    def test_register_webhook_without_secret(self, client, mock_webhook_service):
        """Test sin secreto"""
        response = client.post(
            "/webhooks/register",
            json={
                "url": "https://example.com/webhook",
                "events": ["song.completed"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_register_webhook_error_handling(self, client, mock_webhook_service):
        """Test de manejo de errores"""
        mock_webhook_service.register_webhook.side_effect = Exception("Service error")
        
        response = client.post(
            "/webhooks/register",
            json={
                "url": "https://example.com/webhook",
                "events": ["song.completed"]
            }
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.unit
@pytest.mark.api
class TestListWebhooks:
    """Tests para listar webhooks"""
    
    def test_list_webhooks_success(self, client, mock_webhook_service):
        """Test de listado exitoso"""
        response = client.get("/webhooks/list")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "webhooks" in data or isinstance(data, list)
        if isinstance(data, dict):
            assert len(data.get("webhooks", [])) > 0


@pytest.mark.unit
@pytest.mark.api
class TestGetWebhookStats:
    """Tests para obtener estadísticas de webhooks"""
    
    def test_get_webhook_stats_success(self, client, mock_webhook_service):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/webhooks/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_webhooks" in data or "stats" in data


@pytest.mark.unit
@pytest.mark.api
class TestDeleteWebhook:
    """Tests para eliminar webhook"""
    
    def test_delete_webhook_success(self, client, mock_webhook_service):
        """Test de eliminación exitosa"""
        response = client.delete("/webhooks/webhook-123")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Webhook deleted successfully"
    
    def test_delete_webhook_not_found(self, client, mock_webhook_service):
        """Test cuando el webhook no existe"""
        mock_webhook_service.delete_webhook.return_value = False
        
        response = client.delete("/webhooks/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
@pytest.mark.api
class TestWebhooksIntegration:
    """Tests de integración para webhooks"""
    
    def test_full_webhook_workflow(self, client, mock_webhook_service):
        """Test del flujo completo de webhooks"""
        # 1. Registrar webhook
        register_response = client.post(
            "/webhooks/register",
            json={
                "url": "https://example.com/webhook",
                "events": ["song.completed"]
            }
        )
        assert register_response.status_code == status.HTTP_200_OK
        webhook_id = register_response.json()["webhook_id"]
        
        # 2. Listar webhooks
        list_response = client.get("/webhooks/list")
        assert list_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener estadísticas
        stats_response = client.get("/webhooks/stats")
        assert stats_response.status_code == status.HTTP_200_OK
        
        # 4. Eliminar webhook
        delete_response = client.delete(f"/webhooks/{webhook_id}")
        assert delete_response.status_code == status.HTTP_200_OK




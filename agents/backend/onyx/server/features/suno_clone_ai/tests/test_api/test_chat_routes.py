"""
Tests para las rutas de chat
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.chat import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.get_chat_history = Mock(return_value=[
        {
            "message": "Create a happy song",
            "timestamp": "2024-01-01T00:00:00Z",
            "response": "I'll create a happy song for you"
        },
        {
            "message": "Make it longer",
            "timestamp": "2024-01-01T00:01:00Z",
            "response": "I'll extend the duration"
        }
    ])
    return service


@pytest.fixture
def client(mock_song_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    from api.dependencies import SongServiceDep
    
    app = FastAPI()
    app.include_router(router)
    
    def get_song_service():
        return mock_song_service
    
    app.dependency_overrides[SongServiceDep] = get_song_service
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.api
class TestGetChatHistory:
    """Tests para obtener historial de chat"""
    
    def test_get_chat_history_success(self, client, mock_song_service):
        """Test de obtención exitosa de historial"""
        response = client.get("/chat/history/user-123")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["user_id"] == "user-123"
        assert "history" in data
        assert len(data["history"]) == 2
        assert data["history"][0]["message"] == "Create a happy song"
        
        mock_song_service.get_chat_history.assert_called_once_with(
            "user-123",
            limit=50
        )
    
    def test_get_chat_history_with_limit(self, client, mock_song_service):
        """Test con límite personalizado"""
        response = client.get("/chat/history/user-123?limit=10")
        
        assert response.status_code == status.HTTP_200_OK
        mock_song_service.get_chat_history.assert_called_once_with(
            "user-123",
            limit=10
        )
    
    def test_get_chat_history_limit_validation(self, client):
        """Test de validación de límite"""
        # Límite muy bajo
        response = client.get("/chat/history/user-123?limit=0")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Límite muy alto
        response = client.get("/chat/history/user-123?limit=101")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_chat_history_empty(self, client, mock_song_service):
        """Test cuando no hay historial"""
        mock_song_service.get_chat_history.return_value = []
        
        response = client.get("/chat/history/user-123")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["history"]) == 0
    
    def test_get_chat_history_different_users(self, client, mock_song_service):
        """Test con diferentes usuarios"""
        users = ["user-1", "user-2", "user-3"]
        
        for user_id in users:
            response = client.get(f"/chat/history/{user_id}")
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["user_id"] == user_id


@pytest.mark.integration
@pytest.mark.api
class TestChatIntegration:
    """Tests de integración para chat"""
    
    def test_chat_history_workflow(self, client, mock_song_service):
        """Test del flujo completo de historial"""
        # Obtener historial con diferentes límites
        for limit in [10, 25, 50, 100]:
            response = client.get(f"/chat/history/user-123?limit={limit}")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["history"]) <= limit




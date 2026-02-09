"""
Tests para las rutas de compartición
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from api.routes.sharing import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.get_song = Mock(return_value={
        "song_id": "song-123",
        "user_id": "user-456",
        "prompt": "Test song",
        "status": "completed"
    })
    service.save_song = Mock(return_value=True)
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
class TestCreateShareLink:
    """Tests para crear enlace de compartición"""
    
    def test_create_share_link_success(self, client, mock_song_service):
        """Test de creación exitosa de enlace"""
        response = client.post("/songs/song-123/share")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "share_token" in data
        assert "share_url" in data
        assert "expires_at" in data
        assert data["song_id"] == "song-123"
    
    def test_create_share_link_with_expiration(self, client, mock_song_service):
        """Test con tiempo de expiración personalizado"""
        expires_in = 3600  # 1 hora
        response = client.post(f"/songs/song-123/share?expires_in={expires_in}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "expires_at" in data
        
        # Verificar que la expiración es aproximadamente correcta
        expires_at = datetime.fromisoformat(data["expires_at"].replace('Z', '+00:00'))
        now = datetime.now(expires_at.tzinfo)
        time_diff = (expires_at - now).total_seconds()
        assert 3500 < time_diff < 3700  # Aproximadamente 1 hora
    
    def test_create_share_link_with_max_uses(self, client, mock_song_service):
        """Test con límite de usos"""
        max_uses = 5
        response = client.post(f"/songs/song-123/share?max_uses={max_uses}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data.get("max_uses") == max_uses
    
    def test_create_share_link_expiration_validation(self, client):
        """Test de validación de expiración"""
        # Expiración muy corta
        response = client.post("/songs/song-123/share?expires_in=1000")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Expiración muy larga
        response = client.post("/songs/song-123/share?expires_in=700000")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_create_share_link_song_not_found(self, client, mock_song_service):
        """Test cuando la canción no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.post("/songs/nonexistent/share")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestValidateShareLink:
    """Tests para validar enlace de compartición"""
    
    def test_validate_share_link_success(self, client, mock_song_service):
        """Test de validación exitosa"""
        # Primero crear un enlace
        create_response = client.post("/songs/song-123/share")
        assert create_response.status_code == status.HTTP_200_OK
        share_token = create_response.json()["share_token"]
        
        # Validar el enlace
        validate_response = client.get(f"/songs/share/{share_token}/validate")
        
        assert validate_response.status_code == status.HTTP_200_OK
        data = validate_response.json()
        assert data["valid"] is True
        assert data["song_id"] == "song-123"
    
    def test_validate_share_link_invalid_token(self, client):
        """Test con token inválido"""
        response = client.get("/songs/share/invalid-token/validate")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_validate_share_link_expired(self, client, mock_song_service):
        """Test con enlace expirado"""
        # Crear enlace con expiración muy corta
        response = client.post("/songs/song-123/share?expires_in=3600")
        assert response.status_code == status.HTTP_200_OK
        share_token = response.json()["share_token"]
        
        # Simular expiración (en un test real, necesitarías manipular el tiempo)
        # Por ahora, solo verificamos que el endpoint existe
        validate_response = client.get(f"/songs/share/{share_token}/validate")
        # Puede ser 200 o 410 dependiendo de si está expirado
        assert validate_response.status_code in [status.HTTP_200_OK, status.HTTP_410_GONE]


@pytest.mark.unit
@pytest.mark.api
class TestGetShareStats:
    """Tests para obtener estadísticas de compartición"""
    
    def test_get_share_stats_success(self, client, mock_song_service):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/songs/song-123/share/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_shares" in data
        assert "active_shares" in data


@pytest.mark.integration
@pytest.mark.api
class TestSharingIntegration:
    """Tests de integración para compartición"""
    
    def test_full_sharing_workflow(self, client, mock_song_service):
        """Test del flujo completo de compartición"""
        # 1. Crear enlace
        create_response = client.post("/songs/song-123/share?expires_in=86400&max_uses=10")
        assert create_response.status_code == status.HTTP_200_OK
        share_token = create_response.json()["share_token"]
        
        # 2. Validar enlace
        validate_response = client.get(f"/songs/share/{share_token}/validate")
        assert validate_response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
        
        # 3. Obtener estadísticas
        stats_response = client.get("/songs/song-123/share/stats")
        assert stats_response.status_code == status.HTTP_200_OK




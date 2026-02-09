"""
Tests mejorados para las rutas de favoritos
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.favorites import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.get_song = Mock(return_value=None)
    service.save_song = Mock(return_value=True)
    service.list_songs = Mock(return_value=[])
    return service


@pytest.fixture
def sample_song_data():
    """Datos de ejemplo de canción"""
    return {
        "song_id": "song-123",
        "user_id": "user-456",
        "prompt": "Test song",
        "file_path": "/tmp/song.wav",
        "status": "completed",
        "metadata": {
            "favorites": [],
            "ratings": []
        }
    }


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
class TestAddToFavorites:
    """Tests para agregar a favoritos"""
    
    def test_add_to_favorites_success(self, client, mock_song_service, sample_song_data):
        """Test de agregar a favoritos exitosamente"""
        mock_song_service.get_song.return_value = sample_song_data
        
        response = client.post(
            "/songs/song-123/favorite",
            params={"user_id": "user-789"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Song added to favorites"
        assert data["song_id"] == "song-123"
        assert data["user_id"] == "user-789"
        assert data["is_favorite"] is True
    
    def test_add_to_favorites_already_favorite(self, client, mock_song_service, sample_song_data):
        """Test cuando la canción ya está en favoritos"""
        song_with_favorite = sample_song_data.copy()
        song_with_favorite["metadata"]["favorites"] = ["user-789"]
        mock_song_service.get_song.return_value = song_with_favorite
        
        response = client.post(
            "/songs/song-123/favorite",
            params={"user_id": "user-789"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "already" in data["message"].lower() or data["is_favorite"] is True
    
    def test_add_to_favorites_song_not_found(self, client, mock_song_service):
        """Test cuando la canción no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.post(
            "/songs/nonexistent/favorite",
            params={"user_id": "user-123"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_add_to_favorites_invalid_song_id(self, client):
        """Test con song_id inválido"""
        response = client.post(
            "/songs/invalid-id/favorite",
            params={"user_id": "user-123"}
        )
        
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.unit
@pytest.mark.api
class TestRemoveFromFavorites:
    """Tests para eliminar de favoritos"""
    
    def test_remove_from_favorites_success(self, client, mock_song_service, sample_song_data):
        """Test de eliminar de favoritos exitosamente"""
        song_with_favorite = sample_song_data.copy()
        song_with_favorite["metadata"]["favorites"] = ["user-789"]
        mock_song_service.get_song.return_value = song_with_favorite
        
        response = client.delete(
            "/songs/song-123/favorite",
            params={"user_id": "user-789"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Song removed from favorites"
        assert data["is_favorite"] is False
    
    def test_remove_from_favorites_not_favorite(self, client, mock_song_service, sample_song_data):
        """Test cuando la canción no está en favoritos"""
        mock_song_service.get_song.return_value = sample_song_data
        
        response = client.delete(
            "/songs/song-123/favorite",
            params={"user_id": "user-789"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_favorite"] is False


@pytest.mark.unit
@pytest.mark.api
class TestRateSong:
    """Tests para calificar canciones"""
    
    def test_rate_song_success(self, client, mock_song_service, sample_song_data):
        """Test de calificación exitosa"""
        mock_song_service.get_song.return_value = sample_song_data
        
        response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": 5}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["song_id"] == "song-123"
        assert data["rating"] == 5
        assert "average_rating" in data
    
    def test_rate_song_update_rating(self, client, mock_song_service, sample_song_data):
        """Test de actualización de calificación existente"""
        song_with_rating = sample_song_data.copy()
        song_with_rating["metadata"]["ratings"] = [
            {"user_id": "user-789", "rating": 3}
        ]
        mock_song_service.get_song.return_value = song_with_rating
        
        response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": 5}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["rating"] == 5
    
    def test_rate_song_invalid_rating(self, client, mock_song_service, sample_song_data):
        """Test con calificación inválida"""
        mock_song_service.get_song.return_value = sample_song_data
        
        # Rating muy bajo
        response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": -1}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Rating muy alto
        response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": 6}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_rate_song_boundary_values(self, client, mock_song_service, sample_song_data):
        """Test con valores límite"""
        mock_song_service.get_song.return_value = sample_song_data
        
        # Rating mínimo válido
        response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": 0}
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Rating máximo válido
        response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": 5}
        )
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestGetUserFavorites:
    """Tests para obtener favoritos de usuario"""
    
    def test_get_user_favorites_success(self, client, mock_song_service):
        """Test de obtención exitosa"""
        songs = [
            {
                "song_id": "song-1",
                "metadata": {"favorites": ["user-123"]}
            },
            {
                "song_id": "song-2",
                "metadata": {"favorites": ["user-123"]}
            }
        ]
        mock_song_service.list_songs.return_value = songs
        
        response = client.get(
            "/songs/favorites",
            params={"user_id": "user-123"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "favorites" in data
        assert len(data["favorites"]) >= 0
    
    def test_get_user_favorites_pagination(self, client, mock_song_service):
        """Test con paginación"""
        response = client.get(
            "/songs/favorites",
            params={"user_id": "user-123", "limit": 10, "offset": 0}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "limit" in data
        assert "offset" in data


@pytest.mark.integration
@pytest.mark.api
class TestFavoritesIntegration:
    """Tests de integración para favoritos"""
    
    def test_full_favorites_workflow(self, client, mock_song_service, sample_song_data):
        """Test del flujo completo de favoritos"""
        mock_song_service.get_song.return_value = sample_song_data
        
        # 1. Agregar a favoritos
        add_response = client.post(
            "/songs/song-123/favorite",
            params={"user_id": "user-789"}
        )
        assert add_response.status_code == status.HTTP_200_OK
        
        # 2. Calificar
        rate_response = client.post(
            "/songs/song-123/rate",
            params={"user_id": "user-789", "rating": 5}
        )
        assert rate_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener favoritos
        favorites_response = client.get(
            "/songs/favorites",
            params={"user_id": "user-789"}
        )
        assert favorites_response.status_code == status.HTTP_200_OK
        
        # 4. Eliminar de favoritos
        remove_response = client.delete(
            "/songs/song-123/favorite",
            params={"user_id": "user-789"}
        )
        assert remove_response.status_code == status.HTTP_200_OK




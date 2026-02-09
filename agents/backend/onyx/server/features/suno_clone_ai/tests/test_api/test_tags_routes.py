"""
Tests para las rutas de tags
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.tags import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.get_song = Mock(return_value={
        "song_id": "song-123",
        "user_id": "user-456",
        "prompt": "Test song",
        "status": "completed",
        "metadata": {
            "tags": []
        }
    })
    service.save_song = Mock(return_value=True)
    service.list_songs = Mock(return_value=[])
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
class TestAddTags:
    """Tests para agregar tags"""
    
    def test_add_tags_success(self, client, mock_song_service):
        """Test de agregar tags exitosamente"""
        response = client.post(
            "/songs/song-123/tags",
            params={"tags": "rock,energetic,fast"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Tags added successfully"
        assert "tags" in data
        assert len(data["tags"]) == 3
        assert "rock" in data["tags"]
        assert "energetic" in data["tags"]
        assert "fast" in data["tags"]
    
    def test_add_tags_normalization(self, client, mock_song_service):
        """Test de normalización de tags"""
        response = client.post(
            "/songs/song-123/tags",
            params={"tags": "ROCK, Energetic ,  Fast  "}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Tags deberían estar normalizados (lowercase, sin espacios)
        tags = data["tags"]
        assert all(tag.islower() for tag in tags)
    
    def test_add_tags_duplicates(self, client, mock_song_service):
        """Test que no se dupliquen tags"""
        # Agregar tags iniciales
        first_response = client.post(
            "/songs/song-123/tags",
            params={"tags": "rock,pop"}
        )
        assert first_response.status_code == status.HTTP_200_OK
        
        # Agregar tags duplicados
        second_response = client.post(
            "/songs/song-123/tags",
            params={"tags": "rock,jazz"}
        )
        assert second_response.status_code == status.HTTP_200_OK
        data = second_response.json()
        # "rock" no debería estar duplicado
        assert data["tags"].count("rock") == 1
    
    def test_add_tags_validation(self, client):
        """Test de validación de tags"""
        # Tags vacíos
        response = client.post(
            "/songs/song-123/tags",
            params={"tags": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_add_tags_song_not_found(self, client, mock_song_service):
        """Test cuando la canción no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.post(
            "/songs/nonexistent/tags",
            params={"tags": "rock"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestRemoveTags:
    """Tests para eliminar tags"""
    
    def test_remove_tags_success(self, client, mock_song_service):
        """Test de eliminación exitosa de tags"""
        # Primero agregar tags
        add_response = client.post(
            "/songs/song-123/tags",
            params={"tags": "rock,pop,jazz"}
        )
        assert add_response.status_code == status.HTTP_200_OK
        
        # Eliminar algunos tags
        remove_response = client.delete(
            "/songs/song-123/tags",
            params={"tags": "rock,pop"}
        )
        
        assert remove_response.status_code == status.HTTP_200_OK
        data = remove_response.json()
        assert "jazz" in data["tags"]
        assert "rock" not in data["tags"]
        assert "pop" not in data["tags"]


@pytest.mark.unit
@pytest.mark.api
class TestSearchByTags:
    """Tests para buscar por tags"""
    
    def test_search_by_tags_success(self, client, mock_song_service):
        """Test de búsqueda exitosa por tags"""
        mock_song_service.list_songs.return_value = [
            {
                "song_id": "song-1",
                "metadata": {"tags": ["rock", "energetic"]}
            },
            {
                "song_id": "song-2",
                "metadata": {"tags": ["pop", "happy"]}
            }
        ]
        
        response = client.get("/songs/search/tags?tags=rock")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "songs" in data or isinstance(data, list)
    
    def test_search_by_multiple_tags(self, client, mock_song_service):
        """Test de búsqueda con múltiples tags"""
        response = client.get("/songs/search/tags?tags=rock,energetic")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestGetTagStats:
    """Tests para obtener estadísticas de tags"""
    
    def test_get_tag_stats_success(self, client, mock_song_service):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/songs/tags/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "tags" in data or "popular_tags" in data


@pytest.mark.integration
@pytest.mark.api
class TestTagsIntegration:
    """Tests de integración para tags"""
    
    def test_full_tags_workflow(self, client, mock_song_service):
        """Test del flujo completo de tags"""
        # 1. Agregar tags
        add_response = client.post(
            "/songs/song-123/tags",
            params={"tags": "rock,pop,energetic"}
        )
        assert add_response.status_code == status.HTTP_200_OK
        
        # 2. Buscar por tags
        search_response = client.get("/songs/search/tags?tags=rock")
        assert search_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener estadísticas
        stats_response = client.get("/songs/tags/stats")
        assert stats_response.status_code == status.HTTP_200_OK
        
        # 4. Eliminar tags
        remove_response = client.delete(
            "/songs/song-123/tags",
            params={"tags": "pop"}
        )
        assert remove_response.status_code == status.HTTP_200_OK




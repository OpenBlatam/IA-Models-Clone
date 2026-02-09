"""
Tests para las rutas de gestión de playlists
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime

from api.routes.playlists import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.save_song = Mock(return_value=True)
    service.get_song = Mock(return_value=None)
    service.list_songs = Mock(return_value=[])
    return service


@pytest.fixture
def sample_playlist_data():
    """Datos de ejemplo de playlist"""
    return {
        "playlist_id": "playlist-123",
        "name": "My Favorites",
        "description": "Best songs",
        "user_id": "user-456",
        "is_public": True,
        "songs": ["song-1", "song-2"],
        "song_count": 2,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "play_count": 0,
        "favorites_count": 0
    }


@pytest.fixture
def sample_song_data():
    """Datos de ejemplo de canción"""
    return {
        "song_id": "song-1",
        "user_id": "user-456",
        "prompt": "Test song",
        "file_path": "/tmp/song.wav",
        "status": "completed",
        "metadata": {}
    }


@pytest.fixture
def client(mock_song_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    from api.dependencies import SongServiceDep
    
    app = FastAPI()
    app.include_router(router)
    
    # Mock de la dependencia
    def get_song_service():
        return mock_song_service
    
    app.dependency_overrides[SongServiceDep] = get_song_service
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.api
class TestCreatePlaylist:
    """Tests para el endpoint de creación de playlist"""
    
    def test_create_playlist_success(self, client, mock_song_service):
        """Test de creación exitosa de playlist"""
        response = client.post(
            "/playlists",
            params={
                "name": "My Playlist",
                "description": "Test playlist",
                "user_id": "user-123",
                "is_public": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Playlist created successfully"
        assert "playlist_id" in data
        assert data["playlist"]["name"] == "My Playlist"
        assert data["playlist"]["description"] == "Test playlist"
        assert data["playlist"]["user_id"] == "user-123"
        assert data["playlist"]["is_public"] is True
        assert data["playlist"]["song_count"] == 0
        assert len(data["playlist"]["songs"]) == 0
        
        # Verificar que se guardó
        assert mock_song_service.save_song.called
    
    def test_create_playlist_minimal(self, client, mock_song_service):
        """Test de creación con parámetros mínimos"""
        response = client.post(
            "/playlists",
            params={
                "name": "Minimal Playlist",
                "user_id": "user-123"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["playlist"]["name"] == "Minimal Playlist"
        assert data["playlist"]["description"] is None
        assert data["playlist"]["is_public"] is False
    
    def test_create_playlist_private(self, client, mock_song_service):
        """Test de creación de playlist privada"""
        response = client.post(
            "/playlists",
            params={
                "name": "Private Playlist",
                "user_id": "user-123",
                "is_public": False
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["playlist"]["is_public"] is False
    
    def test_create_playlist_validation_name_too_short(self, client):
        """Test de validación: nombre muy corto"""
        response = client.post(
            "/playlists",
            params={
                "name": "",
                "user_id": "user-123"
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_create_playlist_validation_name_too_long(self, client):
        """Test de validación: nombre muy largo"""
        long_name = "a" * 101
        response = client.post(
            "/playlists",
            params={
                "name": long_name,
                "user_id": "user-123"
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_create_playlist_validation_description_too_long(self, client):
        """Test de validación: descripción muy larga"""
        long_description = "a" * 501
        response = client.post(
            "/playlists",
            params={
                "name": "Test",
                "description": long_description,
                "user_id": "user-123"
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_create_playlist_error_handling(self, client, mock_song_service):
        """Test de manejo de errores"""
        mock_song_service.save_song.side_effect = Exception("Database error")
        
        response = client.post(
            "/playlists",
            params={
                "name": "Test",
                "user_id": "user-123"
            }
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error creating playlist" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestAddSongToPlaylist:
    """Tests para agregar canciones a playlist"""
    
    def test_add_song_success(self, client, mock_song_service, sample_playlist_data, sample_song_data):
        """Test de agregar canción exitosamente"""
        # Configurar mocks
        mock_song_service.get_song.side_effect = lambda song_id: {
            "song-1": sample_song_data,
            f"playlist_{sample_playlist_data['playlist_id']}": {
                "song_id": f"playlist_{sample_playlist_data['playlist_id']}",
                "user_id": sample_playlist_data["user_id"],
                "prompt": f"Playlist: {sample_playlist_data['name']}",
                "file_path": "",
                "metadata": sample_playlist_data
            }
        }.get(song_id)
        
        response = client.post(
            f"/playlists/{sample_playlist_data['playlist_id']}/songs/song-1"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Song added to playlist"
        assert data["playlist_id"] == sample_playlist_data["playlist_id"]
        assert data["song_id"] == "song-1"
        assert data["song_count"] == 3  # 2 originales + 1 nueva
    
    def test_add_song_with_position(self, client, mock_song_service, sample_playlist_data, sample_song_data):
        """Test de agregar canción en posición específica"""
        mock_song_service.get_song.side_effect = lambda song_id: {
            "song-1": sample_song_data,
            f"playlist_{sample_playlist_data['playlist_id']}": {
                "song_id": f"playlist_{sample_playlist_data['playlist_id']}",
                "user_id": sample_playlist_data["user_id"],
                "prompt": f"Playlist: {sample_playlist_data['name']}",
                "file_path": "",
                "metadata": sample_playlist_data
            }
        }.get(song_id)
        
        response = client.post(
            f"/playlists/{sample_playlist_data['playlist_id']}/songs/song-1",
            params={"position": 0}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["position"] == 0
    
    def test_add_song_already_in_playlist(self, client, mock_song_service, sample_playlist_data, sample_song_data):
        """Test de agregar canción que ya está en la playlist"""
        playlist_with_song = sample_playlist_data.copy()
        playlist_with_song["songs"] = ["song-1", "song-2"]
        
        mock_song_service.get_song.side_effect = lambda song_id: {
            "song-1": sample_song_data,
            f"playlist_{playlist_with_song['playlist_id']}": {
                "song_id": f"playlist_{playlist_with_song['playlist_id']}",
                "user_id": playlist_with_song["user_id"],
                "prompt": f"Playlist: {playlist_with_song['name']}",
                "file_path": "",
                "metadata": playlist_with_song
            }
        }.get(song_id)
        
        response = client.post(
            f"/playlists/{playlist_with_song['playlist_id']}/songs/song-1"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Song already in playlist"
        assert data["was_already_in_playlist"] is True
    
    def test_add_song_playlist_not_found(self, client, mock_song_service, sample_song_data):
        """Test cuando la playlist no existe"""
        mock_song_service.get_song.side_effect = lambda song_id: {
            "song-1": sample_song_data
        }.get(song_id)
        
        response = client.post(
            "/playlists/nonexistent/songs/song-1"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    def test_add_song_invalid_song_id(self, client):
        """Test con song_id inválido"""
        response = client.post(
            "/playlists/playlist-123/songs/invalid-id"
        )
        
        # Depende de la validación en validators
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.unit
@pytest.mark.api
class TestRemoveSongFromPlaylist:
    """Tests para eliminar canciones de playlist"""
    
    def test_remove_song_success(self, client, mock_song_service, sample_playlist_data):
        """Test de eliminar canción exitosamente"""
        playlist_with_songs = sample_playlist_data.copy()
        playlist_with_songs["songs"] = ["song-1", "song-2", "song-3"]
        playlist_with_songs["song_count"] = 3
        
        mock_song_service.get_song.return_value = {
            "song_id": f"playlist_{playlist_with_songs['playlist_id']}",
            "user_id": playlist_with_songs["user_id"],
            "prompt": f"Playlist: {playlist_with_songs['name']}",
            "file_path": "",
            "metadata": playlist_with_songs
        }
        
        response = client.delete(
            f"/playlists/{playlist_with_songs['playlist_id']}/songs/song-2"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Song removed from playlist"
        assert data["song_id"] == "song-2"
        assert data["song_count"] == 2
    
    def test_remove_song_not_in_playlist(self, client, mock_song_service, sample_playlist_data):
        """Test de eliminar canción que no está en la playlist"""
        mock_song_service.get_song.return_value = {
            "song_id": f"playlist_{sample_playlist_data['playlist_id']}",
            "user_id": sample_playlist_data["user_id"],
            "prompt": f"Playlist: {sample_playlist_data['name']}",
            "file_path": "",
            "metadata": sample_playlist_data
        }
        
        response = client.delete(
            f"/playlists/{sample_playlist_data['playlist_id']}/songs/nonexistent-song"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["song_count"] == 2  # Sin cambios
    
    def test_remove_song_playlist_not_found(self, client, mock_song_service):
        """Test cuando la playlist no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.delete(
            "/playlists/nonexistent/songs/song-1"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestGetUserPlaylists:
    """Tests para obtener playlists de usuario"""
    
    def test_get_user_playlists_success(self, client, mock_song_service, sample_playlist_data):
        """Test de obtener playlists exitosamente"""
        playlist_song = {
            "song_id": f"playlist_{sample_playlist_data['playlist_id']}",
            "user_id": sample_playlist_data["user_id"],
            "prompt": f"Playlist: {sample_playlist_data['name']}",
            "file_path": "",
            "metadata": sample_playlist_data
        }
        
        mock_song_service.list_songs.return_value = [playlist_song]
        
        response = client.get(
            f"/playlists/users/{sample_playlist_data['user_id']}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["user_id"] == sample_playlist_data["user_id"]
        assert len(data["playlists"]) == 1
        assert data["playlists"][0]["playlist_id"] == sample_playlist_data["playlist_id"]
        assert data["total"] == 1
    
    def test_get_user_playlists_pagination(self, client, mock_song_service, sample_playlist_data):
        """Test de paginación"""
        playlists = []
        for i in range(5):
            playlist = sample_playlist_data.copy()
            playlist["playlist_id"] = f"playlist-{i}"
            playlists.append({
                "song_id": f"playlist_{playlist['playlist_id']}",
                "user_id": playlist["user_id"],
                "prompt": f"Playlist: {playlist['name']}",
                "file_path": "",
                "metadata": playlist
            })
        
        mock_song_service.list_songs.return_value = playlists
        
        response = client.get(
            f"/playlists/users/{sample_playlist_data['user_id']}",
            params={"limit": 2, "offset": 0}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["playlists"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["has_more"] is True
    
    def test_get_user_playlists_empty(self, client, mock_song_service):
        """Test cuando el usuario no tiene playlists"""
        mock_song_service.list_songs.return_value = []
        
        response = client.get(
            "/playlists/users/user-123"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["playlists"]) == 0
        assert data["total"] == 0


@pytest.mark.unit
@pytest.mark.api
class TestGetPlaylist:
    """Tests para obtener información de playlist"""
    
    def test_get_playlist_success(self, client, mock_song_service, sample_playlist_data, sample_song_data):
        """Test de obtener playlist exitosamente"""
        playlist_song = {
            "song_id": f"playlist_{sample_playlist_data['playlist_id']}",
            "user_id": sample_playlist_data["user_id"],
            "prompt": f"Playlist: {sample_playlist_data['name']}",
            "file_path": "",
            "metadata": sample_playlist_data
        }
        
        def get_song_side_effect(song_id):
            if song_id == f"playlist_{sample_playlist_data['playlist_id']}":
                return playlist_song
            elif song_id == "song-1":
                return sample_song_data
            elif song_id == "song-2":
                return {**sample_song_data, "song_id": "song-2"}
            return None
        
        mock_song_service.get_song.side_effect = get_song_side_effect
        
        response = client.get(
            f"/playlists/{sample_playlist_data['playlist_id']}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "playlist" in data
        assert "songs" in data
        assert data["song_count"] == 2
        assert len(data["songs"]) == 2
    
    def test_get_playlist_not_found(self, client, mock_song_service):
        """Test cuando la playlist no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.get(
            "/playlists/nonexistent"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_playlist_with_missing_songs(self, client, mock_song_service, sample_playlist_data):
        """Test cuando algunas canciones no existen"""
        playlist_song = {
            "song_id": f"playlist_{sample_playlist_data['playlist_id']}",
            "user_id": sample_playlist_data["user_id"],
            "prompt": f"Playlist: {sample_playlist_data['name']}",
            "file_path": "",
            "metadata": sample_playlist_data
        }
        
        def get_song_side_effect(song_id):
            if song_id == f"playlist_{sample_playlist_data['playlist_id']}":
                return playlist_song
            return None  # Canciones no encontradas
        
        mock_song_service.get_song.side_effect = get_song_side_effect
        
        response = client.get(
            f"/playlists/{sample_playlist_data['playlist_id']}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["songs"]) == 0  # No se encontraron canciones


@pytest.mark.integration
@pytest.mark.api
class TestPlaylistIntegration:
    """Tests de integración para playlists"""
    
    def test_full_playlist_workflow(self, client, mock_song_service, sample_song_data):
        """Test del flujo completo de gestión de playlist"""
        # 1. Crear playlist
        create_response = client.post(
            "/playlists",
            params={
                "name": "Integration Test Playlist",
                "user_id": "user-123",
                "is_public": True
            }
        )
        assert create_response.status_code == status.HTTP_200_OK
        playlist_id = create_response.json()["playlist_id"]
        
        # 2. Configurar mocks para operaciones posteriores
        playlist_data = {
            "playlist_id": playlist_id,
            "name": "Integration Test Playlist",
            "user_id": "user-123",
            "songs": [],
            "song_count": 0
        }
        
        def get_song_side_effect(song_id):
            if song_id == f"playlist_{playlist_id}":
                return {
                    "song_id": f"playlist_{playlist_id}",
                    "user_id": "user-123",
                    "prompt": f"Playlist: Integration Test Playlist",
                    "file_path": "",
                    "metadata": playlist_data
                }
            elif song_id == "song-1":
                return sample_song_data
            return None
        
        mock_song_service.get_song.side_effect = get_song_side_effect
        
        # 3. Agregar canción
        add_response = client.post(
            f"/playlists/{playlist_id}/songs/song-1"
        )
        assert add_response.status_code == status.HTTP_200_OK
        
        # 4. Obtener playlist
        get_response = client.get(f"/playlists/{playlist_id}")
        assert get_response.status_code == status.HTTP_200_OK
        
        # 5. Eliminar canción
        remove_response = client.delete(
            f"/playlists/{playlist_id}/songs/song-1"
        )
        assert remove_response.status_code == status.HTTP_200_OK




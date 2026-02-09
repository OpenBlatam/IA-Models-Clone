"""
Tests refactorizados para las rutas de playlists
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock
from fastapi import status

from test_helpers import (
    BaseAPITestCase,
    create_router_client,
    assert_standard_response,
    assert_paginated_response
)
from api.routes.playlists import router
from services.playlist_service import PlaylistService


class TestPlaylistsRoutesRefactored(BaseAPITestCase):
    """Tests refactorizados usando BaseAPITestCase"""
    
    router = router
    
    @pytest.fixture
    def mock_playlist_service(self):
        """Mock del servicio de playlists"""
        service = Mock(spec=PlaylistService)
        
        playlist = Mock()
        playlist.playlist_id = "playlist-123"
        playlist.name = "Test Playlist"
        playlist.user_id = "user-123"
        playlist.songs = []
        
        service.create_playlist = Mock(return_value=playlist)
        service.get_playlist = Mock(return_value=playlist)
        service.list_user_playlists = Mock(return_value=[playlist])
        service.add_song = Mock(return_value=True)
        service.remove_song = Mock(return_value=True)
        
        return service
    
    def test_create_playlist_success(self, mock_playlist_service):
        """Test de creación exitosa usando clase base"""
        client = self.create_client({
            "api.routes.playlists.get_playlist_service": mock_playlist_service,
            "api.routes.playlists.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.post(
            "/playlists",
            json={"name": "My Playlist", "description": "Test playlist"}
        )
        
        self.assert_success_response(response)
        self.assert_response_contains_keys(response, ["playlist_id", "name"])
    
    def test_get_playlist_success(self, mock_playlist_service):
        """Test de obtención exitosa"""
        client = self.create_client({
            "api.routes.playlists.get_playlist_service": mock_playlist_service,
            "api.routes.playlists.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.get("/playlists/playlist-123")
        
        self.assert_success_response(response)
        self.assert_response_contains_keys(response, ["playlist_id", "name", "songs"])
    
    def test_add_song_to_playlist(self, mock_playlist_service):
        """Test de agregar canción"""
        client = self.create_client({
            "api.routes.playlists.get_playlist_service": mock_playlist_service,
            "api.routes.playlists.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.post(
            "/playlists/playlist-123/songs",
            json={"song_id": "song-456"}
        )
        
        self.assert_success_response(response)
    
    def test_list_user_playlists(self, mock_playlist_service):
        """Test de listar playlists de usuario"""
        client = self.create_client({
            "api.routes.playlists.get_playlist_service": mock_playlist_service,
            "api.routes.playlists.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.get("/playlists/me")
        
        self.assert_success_response(response)
        # Verificar que es una lista o tiene items
        data = response.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, dict):
            assert "playlists" in data or "items" in data


# Ejemplo usando helpers funcionales
class TestPlaylistsRoutesFunctional:
    """Tests usando helpers funcionales"""
    
    @pytest.fixture
    def mock_playlist_service(self):
        """Mock del servicio"""
        service = Mock(spec=PlaylistService)
        playlist = Mock()
        playlist.playlist_id = "playlist-123"
        playlist.name = "Test"
        service.create_playlist = Mock(return_value=playlist)
        return service
    
    def test_create_playlist_using_helper(self, mock_playlist_service):
        """Test usando create_router_client helper"""
        from test_helpers import create_router_client
        
        client = create_router_client(
            router=router,
            mocks={
                "api.routes.playlists.get_playlist_service": mock_playlist_service,
                "api.routes.playlists.get_current_user": {"user_id": "test_user"}
            }
        )
        
        response = client.post(
            "/playlists",
            json={"name": "My Playlist"}
        )
        
        assert_standard_response(
            response,
            expected_status=200,
            required_keys=["playlist_id", "name"]
        )




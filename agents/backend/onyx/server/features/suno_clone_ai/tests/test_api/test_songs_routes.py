"""
Tests modulares para endpoints de gestión de canciones (routes/songs.py)
"""

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.mock_helpers import create_mock_song_service
from tests.helpers.assertion_helpers import assert_song_response_valid, assert_song_list_valid
from tests.helpers.advanced_helpers import ResponseValidator, MockVerifier


class TestListSongs:
    """Tests para list_songs endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_list_songs_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test exitoso de listado de canciones"""
        songs = [
            create_song_dict(song_id="song-1", user_id="user-1"),
            create_song_dict(song_id="song-2", user_id="user-1"),
            create_song_dict(song_id="song-3", user_id="user-2")
        ]
        mock_song_service.list_songs.return_value = songs
        
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = songs
            
            response = test_client.get(endpoint_path)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "songs" in data
            assert "total" in data
            assert_song_list_valid(data["songs"], min_count=3)
            # Verificar headers de cache
            assert "Cache-Control" in response.headers
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_list_songs_with_pagination(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con paginación"""
        songs = [create_song_dict(song_id=f"song-{i}") for i in range(10)]
        mock_song_service.list_songs.return_value = songs
        
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = songs
            
            response = test_client.get(
                endpoint_path,
                params={"limit": 5, "offset": 0}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["songs"]) <= 5
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_list_songs_filter_by_user(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test filtrando por usuario"""
        user_songs = [
            create_song_dict(song_id="song-1", user_id="user-123"),
            create_song_dict(song_id="song-2", user_id="user-123")
        ]
        mock_song_service.list_songs.return_value = user_songs
        
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = user_songs
            
            response = test_client.get(
                endpoint_path,
                params={"user_id": "user-123"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert all(song.get("user_id") == "user-123" for song in data["songs"])
    
    @pytest.mark.asyncio
    @pytest.mark.boundary
    async def test_list_songs_limit_min(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con límite mínimo"""
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = []
            
            response = test_client.get(
                endpoint_path,
                params={"limit": 1}
            )
            
            assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.asyncio
    @pytest.mark.boundary
    async def test_list_songs_limit_max(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con límite máximo"""
        songs = [create_song_dict(song_id=f"song-{i}") for i in range(100)]
        mock_song_service.list_songs.return_value = songs
        
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = songs
            
            response = test_client.get(
                endpoint_path,
                params={"limit": 100}
            )
            
            assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_list_songs_limit_too_high(
        self,
        test_client,
        endpoint_path
    ):
        """Test con límite excesivo"""
        response = test_client.get(
            endpoint_path,
            params={"limit": 200}  # Excede máximo
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_list_songs_offset_negative(
        self,
        test_client,
        endpoint_path
    ):
        """Test con offset negativo"""
        response = test_client.get(
            endpoint_path,
            params={"offset": -1}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_list_songs_empty_result(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con resultado vacío"""
        mock_song_service.list_songs.return_value = []
        
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = []
            
            response = test_client.get(endpoint_path)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["songs"] == []
            assert data["total"] == 0


class TestGetSong:
    """Tests para get_song endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs/{song_id}"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_get_song_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test exitoso de obtención de canción"""
        song_id = generate_test_song_id()
        song = create_song_dict(song_id=song_id)
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = song
            
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["song_id"] == song_id
            # Verificar headers de cache
            assert "Cache-Control" in response.headers
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_get_song_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        mock_song_service.get_song.return_value = None
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get, \
             patch('api.routes.songs.ensure_song_exists') as mock_ensure:
            mock_get.return_value = None
            from api.routes.songs import SongNotFoundError
            mock_ensure.side_effect = SongNotFoundError(song_id)
            
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_get_song_invalid_id(
        self,
        test_client,
        endpoint_path
    ):
        """Test con ID inválido"""
        invalid_id = "not-a-valid-uuid"
        
        with patch('api.routes.songs.validate_song_id') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid song ID")
            
            response = test_client.get(endpoint_path.format(song_id=invalid_id))
            
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]


class TestDownloadSong:
    """Tests para download_song endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs/{song_id}/download"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_download_song_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service,
        temp_audio_dir
    ):
        """Test exitoso de descarga de canción"""
        from tests.helpers.test_helpers import create_mock_audio, save_test_audio
        
        song_id = generate_test_song_id()
        audio_file = temp_audio_dir / f"{song_id}.wav"
        audio = create_mock_audio()
        save_test_audio(audio, audio_file)
        
        song = create_song_dict(song_id=song_id, file_path=str(audio_file))
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get, \
             patch('api.routes.songs.ensure_song_exists') as mock_ensure, \
             patch('api.routes.songs.ensure_audio_file_exists') as mock_ensure_file, \
             patch('api.routes.songs.get_audio_file_path') as mock_path, \
             patch('api.routes.songs.get_media_type_from_path') as mock_media:
            
            mock_get.return_value = song
            mock_ensure.return_value = song
            mock_ensure_file.return_value = audio_file
            mock_path.return_value = audio_file
            mock_media.return_value = "audio/wav"
            
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_download_song_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        mock_song_service.get_song.return_value = None
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get, \
             patch('api.routes.songs.ensure_song_exists') as mock_ensure:
            mock_get.return_value = None
            from api.routes.songs import SongNotFoundError
            mock_ensure.side_effect = SongNotFoundError(song_id)
            
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_download_song_file_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando el archivo de audio no existe"""
        song_id = generate_test_song_id()
        song = create_song_dict(song_id=song_id, file_path="/nonexistent/file.wav")
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get, \
             patch('api.routes.songs.ensure_song_exists') as mock_ensure, \
             patch('api.routes.songs.ensure_audio_file_exists') as mock_ensure_file:
            mock_get.return_value = song
            mock_ensure.return_value = song
            from api.routes.songs import AudioFileNotFoundError
            mock_ensure_file.side_effect = AudioFileNotFoundError(song_id)
            
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteSong:
    """Tests para delete_song endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs/{song_id}"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_delete_song_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test exitoso de eliminación de canción"""
        song_id = generate_test_song_id()
        mock_song_service.delete_song.return_value = True
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = True
            
            response = test_client.delete(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_204_NO_CONTENT
            MockVerifier.verify_call_count(mock_song_service, 1, "delete_song")
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_delete_song_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        mock_song_service.delete_song.return_value = False
        
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = False
            
            response = test_client.delete(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestSongsIntegration:
    """Tests de integración para endpoints de canciones"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_crud_flow(
        self,
        test_client,
        mock_song_service,
        temp_audio_dir
    ):
        """Test del flujo completo CRUD"""
        from tests.helpers.test_helpers import create_mock_audio, save_test_audio
        
        # 1. Listar canciones (vacío inicialmente)
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = []
            
            list_response = test_client.get("/suno/songs")
            assert list_response.status_code == status.HTTP_200_OK
            assert len(list_response.json()["songs"]) == 0
        
        # 2. Crear canción (simulado)
        song_id = generate_test_song_id()
        song = create_song_dict(song_id=song_id)
        
        # 3. Obtener canción
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get, \
             patch('api.routes.songs.ensure_song_exists') as mock_ensure:
            mock_get.return_value = song
            mock_ensure.return_value = song
            
            get_response = test_client.get(f"/suno/songs/{song_id}")
            assert get_response.status_code == status.HTTP_200_OK
            assert get_response.json()["song_id"] == song_id
        
        # 4. Eliminar canción
        mock_song_service.delete_song.return_value = True
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = True
            
            delete_response = test_client.delete(f"/suno/songs/{song_id}")
            assert delete_response.status_code == status.HTTP_204_NO_CONTENT


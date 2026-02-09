"""
Tests modulares para endpoints de gestión de canciones
"""

import pytest
from fastapi import status
from pathlib import Path
from unittest.mock import patch
from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.mock_helpers import create_mock_song_service
from tests.helpers.assertion_helpers import assert_song_response_valid, assert_song_list_valid


class TestListSongs:
    """Tests para el endpoint list_songs"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs"
    
    @pytest.mark.asyncio
    async def test_list_songs_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test exitoso de listado de canciones"""
        songs = [
            create_song_dict(song_id="song-1"),
            create_song_dict(song_id="song-2")
        ]
        mock_song_service.list_songs.return_value = songs
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(endpoint_path)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "songs" in data
            assert "total" in data
            assert_song_list_valid(data["songs"], min_count=2)
    
    @pytest.mark.asyncio
    async def test_list_songs_with_pagination(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con paginación"""
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(
                endpoint_path,
                params={"limit": 10, "offset": 0}
            )
            
            assert response.status_code == status.HTTP_200_OK
            mock_song_service.list_songs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_songs_filter_by_user(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test filtrando por usuario"""
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(
                endpoint_path,
                params={"user_id": "test-user-123"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            mock_song_service.list_songs.assert_called_with(
                user_id="test-user-123",
                limit=50,
                offset=0
            )


class TestGetSong:
    """Tests para el endpoint get_song"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs/{song_id}"
    
    @pytest.mark.asyncio
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
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["song_id"] == song_id
    
    @pytest.mark.asyncio
    async def test_get_song_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        mock_song_service.get_song.return_value = None
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDownloadSong:
    """Tests para el endpoint download_song"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs/{song_id}/download"
    
    @pytest.mark.asyncio
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
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
    
    @pytest.mark.asyncio
    async def test_download_song_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        mock_song_service.get_song.return_value = None
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.get(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteSong:
    """Tests para el endpoint delete_song"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/songs/{song_id}"
    
    @pytest.mark.asyncio
    async def test_delete_song_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test exitoso de eliminación de canción"""
        song_id = generate_test_song_id()
        mock_song_service.delete_song.return_value = True
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.delete(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_204_NO_CONTENT
            mock_song_service.delete_song.assert_called_once_with(song_id)
    
    @pytest.mark.asyncio
    async def test_delete_song_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        mock_song_service.delete_song.return_value = False
        
        with patch('api.dependencies.get_song_service', return_value=mock_song_service):
            response = test_client.delete(endpoint_path.format(song_id=song_id))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


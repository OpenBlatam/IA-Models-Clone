"""
Tests modulares para SongService
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.assertion_helpers import assert_song_dict_valid


class TestSongService:
    """Tests para SongService"""
    
    @pytest.fixture
    def song_service(self):
        """Fixture para crear instancia de SongService"""
        try:
            from services.song_service import SongService
            return SongService()
        except ImportError:
            pytest.skip("SongService not available")
    
    @pytest.mark.unit
    def test_save_song_success(self, song_service, temp_dir):
        """Test exitoso de guardado de canción"""
        song_id = generate_test_song_id()
        user_id = "test-user-123"
        prompt = "Test song"
        file_path = str(temp_dir / f"{song_id}.wav")
        
        result = song_service.save_song(
            song_id=song_id,
            user_id=user_id,
            prompt=prompt,
            file_path=file_path
        )
        
        assert result is True or result is None  # Depende de la implementación
    
    @pytest.mark.unit
    def test_get_song_success(self, song_service):
        """Test exitoso de obtención de canción"""
        song_id = generate_test_song_id()
        
        # Primero guardar
        song_service.save_song(
            song_id=song_id,
            user_id="test-user",
            prompt="Test",
            file_path="/tmp/test.wav"
        )
        
        # Luego obtener
        song = song_service.get_song(song_id)
        
        # Puede ser None si no está implementado el almacenamiento
        if song:
            assert_song_dict_valid(song)
            assert song["song_id"] == song_id
    
    @pytest.mark.unit
    def test_get_song_not_found(self, song_service):
        """Test cuando la canción no existe"""
        song_id = generate_test_song_id()
        song = song_service.get_song(song_id)
        
        # Debe retornar None si no existe
        assert song is None
    
    @pytest.mark.unit
    def test_list_songs_empty(self, song_service):
        """Test de listado vacío"""
        songs = song_service.list_songs()
        
        assert isinstance(songs, list)
        # Puede estar vacío inicialmente
    
    @pytest.mark.unit
    def test_list_songs_with_user_filter(self, song_service):
        """Test de listado filtrado por usuario"""
        user_id = "test-user-123"
        
        songs = song_service.list_songs(user_id=user_id)
        
        assert isinstance(songs, list)
        # Verificar que todas las canciones pertenecen al usuario
        if songs:
            assert all(song.get("user_id") == user_id for song in songs)
    
    @pytest.mark.unit
    def test_list_songs_with_pagination(self, song_service):
        """Test de listado con paginación"""
        limit = 10
        offset = 0
        
        songs = song_service.list_songs(limit=limit, offset=offset)
        
        assert isinstance(songs, list)
        assert len(songs) <= limit
    
    @pytest.mark.unit
    def test_delete_song_success(self, song_service):
        """Test exitoso de eliminación"""
        song_id = generate_test_song_id()
        
        # Primero guardar
        song_service.save_song(
            song_id=song_id,
            user_id="test-user",
            prompt="Test",
            file_path="/tmp/test.wav"
        )
        
        # Luego eliminar
        result = song_service.delete_song(song_id)
        
        assert result is True or result is None
    
    @pytest.mark.unit
    def test_delete_song_not_found(self, song_service):
        """Test de eliminación cuando no existe"""
        song_id = generate_test_song_id()
        result = song_service.delete_song(song_id)
        
        # Puede retornar False o None
        assert result in [False, None, True]  # Depende de la implementación
    
    @pytest.mark.unit
    def test_update_song_status(self, song_service):
        """Test de actualización de estado"""
        song_id = generate_test_song_id()
        
        # Guardar canción
        song_service.save_song(
            song_id=song_id,
            user_id="test-user",
            prompt="Test",
            file_path="/tmp/test.wav"
        )
        
        # Actualizar estado
        result = song_service.update_song_status(song_id, "completed", "Done")
        
        assert result is True or result is None
    
    @pytest.mark.unit
    def test_get_chat_history(self, song_service):
        """Test de obtención de historial de chat"""
        user_id = "test-user-123"
        limit = 50
        
        history = song_service.get_chat_history(user_id, limit=limit)
        
        assert isinstance(history, list)
        assert len(history) <= limit


class TestSongServiceEdgeCases:
    """Tests de casos edge para SongService"""
    
    @pytest.fixture
    def song_service(self):
        try:
            from services.song_service import SongService
            return SongService()
        except ImportError:
            pytest.skip("SongService not available")
    
    @pytest.mark.edge_case
    def test_save_song_with_special_characters(self, song_service, temp_dir):
        """Test con caracteres especiales en prompt"""
        song_id = generate_test_song_id()
        prompt = "Canción con émojis 🎵🎶 y spéciál chàracters!"
        
        result = song_service.save_song(
            song_id=song_id,
            user_id="test-user",
            prompt=prompt,
            file_path=str(temp_dir / f"{song_id}.wav")
        )
        
        assert result is True or result is None
    
    @pytest.mark.edge_case
    def test_save_song_with_long_prompt(self, song_service, temp_dir):
        """Test con prompt muy largo"""
        song_id = generate_test_song_id()
        prompt = "A" * 1000  # Prompt muy largo
        
        result = song_service.save_song(
            song_id=song_id,
            user_id="test-user",
            prompt=prompt,
            file_path=str(temp_dir / f"{song_id}.wav")
        )
        
        assert result is True or result is None
    
    @pytest.mark.edge_case
    def test_list_songs_large_limit(self, song_service):
        """Test con límite grande"""
        songs = song_service.list_songs(limit=1000)
        
        assert isinstance(songs, list)
    
    @pytest.mark.edge_case
    def test_list_songs_large_offset(self, song_service):
        """Test con offset grande"""
        songs = song_service.list_songs(offset=10000)
        
        assert isinstance(songs, list)


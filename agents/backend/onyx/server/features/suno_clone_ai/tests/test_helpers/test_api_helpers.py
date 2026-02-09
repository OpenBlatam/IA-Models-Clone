"""
Tests modulares para helpers de API
"""

import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import soundfile as sf

from api.helpers import (
    generate_song_id,
    ensure_storage_dir,
    get_audio_file_path,
    load_audio_file,
    save_audio_file,
    create_song_info_from_request,
    notify_song_started,
    notify_song_completed,
    notify_song_failed,
    apply_audio_operations
)
from tests.helpers.test_helpers import create_mock_audio
from tests.helpers.assertion_helpers import assert_audio_valid


class TestGenerateSongId:
    """Tests para generate_song_id"""
    
    def test_generate_song_id_returns_string(self):
        """Test que retorna un string"""
        song_id = generate_song_id()
        assert isinstance(song_id, str)
        assert len(song_id) > 0
    
    def test_generate_song_id_unique(self):
        """Test que genera IDs únicos"""
        ids = {generate_song_id() for _ in range(100)}
        assert len(ids) == 100
    
    def test_generate_song_id_format(self):
        """Test que el formato es válido UUID"""
        song_id = generate_song_id()
        # Debe ser un UUID válido
        uuid.UUID(song_id)


class TestEnsureStorageDir:
    """Tests para ensure_storage_dir"""
    
    def test_ensure_storage_dir_creates_directory(self, temp_dir, monkeypatch):
        """Test que crea el directorio si no existe"""
        storage_path = temp_dir / "storage"
        monkeypatch.setattr("config.settings.audio_storage_path", str(storage_path))
        
        result = ensure_storage_dir()
        
        assert result.exists()
        assert result.is_dir()
        assert result == storage_path
    
    def test_ensure_storage_dir_existing_directory(self, temp_dir, monkeypatch):
        """Test con directorio existente"""
        storage_path = temp_dir / "storage"
        storage_path.mkdir(parents=True)
        monkeypatch.setattr("config.settings.audio_storage_path", str(storage_path))
        
        result = ensure_storage_dir()
        
        assert result.exists()
        assert result == storage_path


class TestGetAudioFilePath:
    """Tests para get_audio_file_path"""
    
    def test_get_audio_file_path_default_extension(self, temp_dir, monkeypatch):
        """Test con extensión por defecto"""
        storage_path = temp_dir / "storage"
        storage_path.mkdir(parents=True)
        monkeypatch.setattr("config.settings.audio_storage_path", str(storage_path))
        
        song_id = "test-song-123"
        file_path = get_audio_file_path(song_id)
        
        assert file_path.name == f"{song_id}.wav"
        assert file_path.parent == storage_path
    
    def test_get_audio_file_path_custom_extension(self, temp_dir, monkeypatch):
        """Test con extensión personalizada"""
        storage_path = temp_dir / "storage"
        storage_path.mkdir(parents=True)
        monkeypatch.setattr("config.settings.audio_storage_path", str(storage_path))
        
        song_id = "test-song-123"
        file_path = get_audio_file_path(song_id, extension="mp3")
        
        assert file_path.name == f"{song_id}.mp3"
        assert file_path.suffix == ".mp3"


class TestLoadAudioFile:
    """Tests para load_audio_file"""
    
    def test_load_audio_file_success(self, temp_dir):
        """Test exitoso de carga de audio"""
        audio_file = temp_dir / "test.wav"
        audio = create_mock_audio()
        sf.write(str(audio_file), audio, 44100)
        
        loaded_audio, sample_rate = load_audio_file(str(audio_file))
        
        assert_audio_valid(loaded_audio)
        assert sample_rate == 44100
    
    def test_load_audio_file_not_found(self):
        """Test cuando el archivo no existe"""
        with pytest.raises(FileNotFoundError):
            load_audio_file("/nonexistent/file.wav")


class TestSaveAudioFile:
    """Tests para save_audio_file"""
    
    def test_save_audio_file_success(self, temp_dir):
        """Test exitoso de guardado de audio"""
        audio_file = temp_dir / "test.wav"
        audio = create_mock_audio()
        
        save_audio_file(audio, audio_file, 44100)
        
        assert audio_file.exists()
        assert_audio_valid(np.array(sf.read(str(audio_file))[0]))
    
    def test_save_audio_file_creates_directory(self, temp_dir):
        """Test que crea el directorio si no existe"""
        audio_file = temp_dir / "subdir" / "test.wav"
        audio = create_mock_audio()
        
        save_audio_file(audio, audio_file, 44100)
        
        assert audio_file.exists()
        assert audio_file.parent.exists()


class TestCreateSongInfoFromRequest:
    """Tests para create_song_info_from_request"""
    
    def test_create_song_info_from_request_complete(self):
        """Test con request completo"""
        request = Mock()
        request.prompt = "A happy song"
        request.genre = "pop"
        request.mood = "happy"
        request.duration = 30
        
        song_info = create_song_info_from_request(request)
        
        assert song_info["prompt"] == "A happy song"
        assert song_info["genre"] == "pop"
        assert song_info["mood"] == "happy"
        assert song_info["duration"] == 30
        assert "instruments" in song_info


class TestNotifySongStarted:
    """Tests para notify_song_started"""
    
    @pytest.mark.asyncio
    async def test_notify_song_started_success(self):
        """Test exitoso de notificación"""
        notification_service = AsyncMock()
        user_id = "test-user-123"
        song_id = "test-song-456"
        
        await notify_song_started(notification_service, user_id, song_id)
        
        notification_service.notify_generation_started.assert_called_once_with(
            user_id, song_id
        )
    
    @pytest.mark.asyncio
    async def test_notify_song_started_no_user_id(self):
        """Test sin user_id"""
        notification_service = AsyncMock()
        
        await notify_song_started(notification_service, None, "test-song-456")
        
        notification_service.notify_generation_started.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_notify_song_started_no_service(self):
        """Test sin servicio de notificaciones"""
        await notify_song_started(None, "test-user-123", "test-song-456")
        # No debe lanzar excepción


class TestNotifySongCompleted:
    """Tests para notify_song_completed"""
    
    @pytest.mark.asyncio
    async def test_notify_song_completed_success(self):
        """Test exitoso de notificación"""
        notification_service = AsyncMock()
        user_id = "test-user-123"
        song_id = "test-song-456"
        audio_url = "/songs/test-song-456/download"
        
        await notify_song_completed(notification_service, user_id, song_id, audio_url)
        
        notification_service.notify_song_completed.assert_called_once_with(
            user_id, song_id, audio_url
        )


class TestNotifySongFailed:
    """Tests para notify_song_failed"""
    
    @pytest.mark.asyncio
    async def test_notify_song_failed_success(self):
        """Test exitoso de notificación de error"""
        notification_service = AsyncMock()
        user_id = "test-user-123"
        song_id = "test-song-456"
        error_message = "Generation failed"
        
        await notify_song_failed(notification_service, user_id, song_id, error_message)
        
        notification_service.notify_song_failed.assert_called_once_with(
            user_id, song_id, error_message
        )


class TestApplyAudioOperations:
    """Tests para apply_audio_operations"""
    
    def test_apply_audio_operations_normalize(self):
        """Test con normalización"""
        audio = create_mock_audio()
        audio_processor = Mock()
        audio_processor.normalize = Mock(side_effect=lambda x: x)
        
        result = apply_audio_operations(
            audio=audio,
            audio_processor=audio_processor,
            operations=[],
            normalize=True
        )
        
        audio_processor.normalize.assert_called_once()
        assert result is not None
    
    def test_apply_audio_operations_trim_silence(self):
        """Test con trim de silencio"""
        audio = create_mock_audio()
        audio_processor = Mock()
        audio_processor.trim_silence = Mock(side_effect=lambda x: x)
        audio_processor.normalize = Mock(side_effect=lambda x: x)
        
        result = apply_audio_operations(
            audio=audio,
            audio_processor=audio_processor,
            operations=[],
            trim_silence=True
        )
        
        audio_processor.trim_silence.assert_called_once()
    
    def test_apply_audio_operations_reverb(self):
        """Test con efecto reverb"""
        audio = create_mock_audio()
        audio_processor = Mock()
        audio_processor.normalize = Mock(side_effect=lambda x: x)
        audio_processor.apply_reverb = Mock(side_effect=lambda x, **kwargs: x)
        
        operations = [{"type": "reverb", "room_size": 0.5, "damping": 0.5}]
        
        result = apply_audio_operations(
            audio=audio,
            audio_processor=audio_processor,
            operations=operations
        )
        
        audio_processor.apply_reverb.assert_called_once()
    
    def test_apply_audio_operations_eq(self):
        """Test con EQ"""
        audio = create_mock_audio()
        audio_processor = Mock()
        audio_processor.normalize = Mock(side_effect=lambda x: x)
        audio_processor.apply_eq = Mock(side_effect=lambda x, **kwargs: x)
        
        operations = [{"type": "eq", "low_gain": 1.0, "mid_gain": 0.5, "high_gain": 0.0}]
        
        result = apply_audio_operations(
            audio=audio,
            audio_processor=audio_processor,
            operations=operations
        )
        
        audio_processor.apply_eq.assert_called_once()
    
    def test_apply_audio_operations_fade(self):
        """Test con fade in/out"""
        audio = create_mock_audio()
        audio_processor = Mock()
        audio_processor.normalize = Mock(side_effect=lambda x: x)
        audio_processor.apply_fade = Mock(side_effect=lambda x, **kwargs: x)
        
        result = apply_audio_operations(
            audio=audio,
            audio_processor=audio_processor,
            operations=[],
            fade_in=0.5,
            fade_out=0.5
        )
        
        audio_processor.apply_fade.assert_called_once()


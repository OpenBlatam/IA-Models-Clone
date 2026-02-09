"""
Tests para las rutas de generación de letras
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.lyrics import router
from services.lyrics_generator import LyricsGenerator
from services.audio_transcription import TranscriptionService


@pytest.fixture
def mock_lyrics_generator():
    """Mock del generador de letras"""
    generator = Mock(spec=LyricsGenerator)
    lyrics_obj = Mock()
    lyrics_obj.title = "Test Song"
    lyrics_obj.verses = ["Verse 1", "Verse 2", "Verse 3"]
    lyrics_obj.chorus = "Chorus line"
    lyrics_obj.bridge = "Bridge line"
    lyrics_obj.language = "en"
    lyrics_obj.style = "pop"
    lyrics_obj.theme = "love"
    
    generator.generate_lyrics = Mock(return_value=lyrics_obj)
    generator.generate_from_music = Mock(return_value=lyrics_obj)
    return generator


@pytest.fixture
def mock_transcription_service():
    """Mock del servicio de transcripción"""
    service = Mock(spec=TranscriptionService)
    service.transcribe = Mock(return_value="Transcribed text")
    return service


@pytest.fixture
def client(mock_lyrics_generator, mock_transcription_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.lyrics.get_lyrics_generator', return_value=mock_lyrics_generator):
        with patch('api.routes.lyrics.get_transcription_service', return_value=mock_transcription_service):
            with patch('api.routes.lyrics.get_current_user', return_value={"user_id": "test_user"}):
                yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestGenerateLyrics:
    """Tests para el endpoint de generación de letras"""
    
    def test_generate_lyrics_success(self, client, mock_lyrics_generator):
        """Test de generación exitosa de letras"""
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "love",
                "style": "pop",
                "language": "en",
                "num_verses": 3,
                "include_chorus": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["title"] == "Test Song"
        assert len(data["verses"]) == 3
        assert data["chorus"] == "Chorus line"
        assert data["language"] == "en"
        assert data["style"] == "pop"
        assert data["theme"] == "love"
        
        mock_lyrics_generator.generate_lyrics.assert_called_once_with(
            theme="love",
            style="pop",
            language="en",
            num_verses=3,
            include_chorus=True
        )
    
    def test_generate_lyrics_default_values(self, client, mock_lyrics_generator):
        """Test con valores por defecto"""
        response = client.post(
            "/lyrics/generate",
            json={"theme": "summer"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_lyrics_generator.generate_lyrics.assert_called_once_with(
            theme="summer",
            style=None,
            language="en",
            num_verses=3,
            include_chorus=True
        )
    
    def test_generate_lyrics_min_verses(self, client, mock_lyrics_generator):
        """Test con número mínimo de versos"""
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "test",
                "num_verses": 1
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_lyrics_generator.generate_lyrics.assert_called_once()
        call_args = mock_lyrics_generator.generate_lyrics.call_args[1]
        assert call_args["num_verses"] == 1
    
    def test_generate_lyrics_max_verses(self, client, mock_lyrics_generator):
        """Test con número máximo de versos"""
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "test",
                "num_verses": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_lyrics_generator.generate_lyrics.call_args[1]
        assert call_args["num_verses"] == 10
    
    def test_generate_lyrics_without_chorus(self, client, mock_lyrics_generator):
        """Test sin coro"""
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "test",
                "include_chorus": False
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_lyrics_generator.generate_lyrics.call_args[1]
        assert call_args["include_chorus"] is False
    
    def test_generate_lyrics_error_handling(self, client, mock_lyrics_generator):
        """Test de manejo de errores"""
        mock_lyrics_generator.generate_lyrics.side_effect = Exception("Generation failed")
        
        response = client.post(
            "/lyrics/generate",
            json={"theme": "test"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error generating lyrics" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestGenerateLyricsFromAudio:
    """Tests para el endpoint de generación de letras desde audio"""
    
    def test_generate_from_audio_success(self, client, mock_lyrics_generator, mock_transcription_service):
        """Test de generación exitosa desde audio"""
        audio_content = b"fake audio content"
        audio_file = io.BytesIO(audio_content)
        audio_file.name = "test_audio.wav"
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test_audio.wav"
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_file
            
            with patch('os.path.splitext', return_value=("test_audio", ".wav")):
                with patch('os.unlink'):
                    response = client.post(
                        "/lyrics/generate-from-audio",
                        files={"file": ("test_audio.wav", audio_file, "audio/wav")}
                    )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["title"] == "Test Song"
            assert len(data["verses"]) == 3
    
    def test_generate_from_audio_invalid_file(self, client):
        """Test con archivo inválido"""
        response = client.post(
            "/lyrics/generate-from-audio",
            files={"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")}
        )
        
        # Debería fallar al procesar el archivo
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_generate_from_audio_error_handling(self, client, mock_lyrics_generator):
        """Test de manejo de errores en generación desde audio"""
        mock_lyrics_generator.generate_from_music.side_effect = Exception("Processing failed")
        
        audio_file = io.BytesIO(b"fake audio")
        audio_file.name = "test.wav"
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test.wav"
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_file
            
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    response = client.post(
                        "/lyrics/generate-from-audio",
                        files={"file": ("test.wav", audio_file, "audio/wav")}
                    )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Error generating lyrics from audio" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestLyricsValidation:
    """Tests de validación de parámetros"""
    
    def test_invalid_num_verses_too_low(self, client):
        """Test con número de versos muy bajo"""
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "test",
                "num_verses": 0
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_num_verses_too_high(self, client):
        """Test con número de versos muy alto"""
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "test",
                "num_verses": 11
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_theme(self, client):
        """Test sin tema requerido"""
        response = client.post(
            "/lyrics/generate",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.unit
@pytest.mark.api
class TestLyricsEdgeCases:
    """Tests de casos edge para letras"""
    
    def test_generate_lyrics_empty_theme(self, client):
        """Test con tema vacío"""
        response = client.post(
            "/lyrics/generate",
            json={"theme": ""}
        )
        
        # Puede ser válido o inválido dependiendo de la validación
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_generate_lyrics_very_long_theme(self, client):
        """Test con tema muy largo"""
        long_theme = "a" * 1000
        response = client.post(
            "/lyrics/generate",
            json={"theme": long_theme}
        )
        
        # Debería manejar temas largos
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_generate_lyrics_special_characters(self, client, mock_lyrics_generator):
        """Test con caracteres especiales en el tema"""
        special_theme = "¡Hola! ¿Cómo estás? 中文 日本語 🎵"
        response = client.post(
            "/lyrics/generate",
            json={"theme": special_theme}
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_generate_lyrics_unicode(self, client, mock_lyrics_generator):
        """Test con caracteres unicode"""
        unicode_theme = "Amor y pasión 💕"
        response = client.post(
            "/lyrics/generate",
            json={"theme": unicode_theme}
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_generate_lyrics_different_languages(self, client, mock_lyrics_generator):
        """Test con diferentes idiomas"""
        languages = ["en", "es", "fr", "de", "it"]
        
        for lang in languages:
            response = client.post(
                "/lyrics/generate",
                json={"theme": "test", "language": lang}
            )
            assert response.status_code == status.HTTP_200_OK
            call_args = mock_lyrics_generator.generate_lyrics.call_args[1]
            assert call_args["language"] == lang


@pytest.mark.integration
@pytest.mark.api
class TestLyricsIntegration:
    """Tests de integración para letras"""
    
    def test_full_lyrics_generation_flow(self, client, mock_lyrics_generator):
        """Test del flujo completo de generación"""
        # Generar letras
        response = client.post(
            "/lyrics/generate",
            json={
                "theme": "adventure",
                "style": "rock",
                "language": "en",
                "num_verses": 4,
                "include_chorus": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verificar estructura completa
        assert "title" in data
        assert "verses" in data
        assert "chorus" in data
        assert "bridge" in data
        assert "language" in data
        assert "style" in data
        assert "theme" in data
        
        # Verificar que se llamó al generador
        assert mock_lyrics_generator.generate_lyrics.called
    
    def test_lyrics_generation_and_audio_workflow(self, client, mock_lyrics_generator, sample_audio_file):
        """Test del flujo completo: generar letras y luego desde audio"""
        # 1. Generar letras
        generate_response = client.post(
            "/lyrics/generate",
            json={"theme": "test song"}
        )
        assert generate_response.status_code == status.HTTP_200_OK
        
        # 2. Generar letras desde audio
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    audio_response = client.post(
                        "/lyrics/generate-from-audio",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert audio_response.status_code == status.HTTP_200_OK


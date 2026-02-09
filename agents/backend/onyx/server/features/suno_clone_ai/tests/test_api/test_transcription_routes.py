"""
Tests para las rutas de transcripción
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.transcription import router
from services.audio_transcription import TranscriptionService


@pytest.fixture
def mock_transcription_service():
    """Mock del servicio de transcripción"""
    service = Mock(spec=TranscriptionService)
    
    # Mock de resultado de transcripción
    transcription_result = Mock()
    transcription_result.text = "This is a test transcription"
    transcription_result.language = "en"
    transcription_result.duration = 10.5
    transcription_result.confidence = 0.95
    transcription_result.segments = [
        Mock(start=0.0, end=5.0, text="This is a", confidence=0.9),
        Mock(start=5.0, end=10.5, text="test transcription", confidence=1.0)
    ]
    
    service.transcribe = Mock(return_value=transcription_result)
    
    # Mock de detección de idioma
    language_result = Mock()
    language_result.language = "en"
    language_result.confidence = 0.98
    service.detect_language = Mock(return_value=language_result)
    
    # Mock de resumen
    summary_result = Mock()
    summary_result.summary = "Test summary"
    summary_result.key_points = ["point1", "point2"]
    service.summarize = Mock(return_value=summary_result)
    
    return service


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    return io.BytesIO(b"fake audio content for transcription testing")


@pytest.fixture
def client(mock_transcription_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.transcription.get_transcription_service', return_value=mock_transcription_service):
        with patch('api.routes.transcription.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestTranscribeAudio:
    """Tests para el endpoint de transcripción"""
    
    def test_transcribe_audio_success(self, client, mock_transcription_service, sample_audio_file):
        """Test de transcripción exitosa"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/transcription/transcribe",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "text" in data
        assert data["text"] == "This is a test transcription"
        assert data["language"] == "en"
        assert data["duration"] == 10.5
        assert data["confidence"] == 0.95
        assert "segments" in data
        assert len(data["segments"]) == 2
    
    def test_transcribe_audio_with_language(self, client, mock_transcription_service, sample_audio_file):
        """Test de transcripción con idioma especificado"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/transcription/transcribe",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                        params={"language": "es"}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        mock_transcription_service.transcribe.assert_called()
        call_args = mock_transcription_service.transcribe.call_args
        assert call_args[1].get("language") == "es"
    
    def test_transcribe_audio_error_handling(self, client, mock_transcription_service, sample_audio_file):
        """Test de manejo de errores"""
        mock_transcription_service.transcribe.side_effect = Exception("Transcription failed")
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/transcription/transcribe",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error transcribing audio" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestDetectLanguage:
    """Tests para el endpoint de detección de idioma"""
    
    def test_detect_language_success(self, client, mock_transcription_service, sample_audio_file):
        """Test de detección exitosa de idioma"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/transcription/detect-language",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "language" in data
        assert data["language"] == "en"
        assert "confidence" in data
        assert data["confidence"] == 0.98


@pytest.mark.unit
@pytest.mark.api
class TestSummarizeTranscription:
    """Tests para el endpoint de resumen"""
    
    def test_summarize_success(self, client, mock_transcription_service):
        """Test de resumen exitoso"""
        response = client.post(
            "/transcription/summarize",
            json={"text": "This is a long transcription text that needs to be summarized"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "summary" in data
        assert data["summary"] == "Test summary"
        assert "key_points" in data
        assert len(data["key_points"]) == 2


@pytest.mark.integration
@pytest.mark.api
class TestTranscriptionIntegration:
    """Tests de integración para transcripción"""
    
    def test_full_transcription_workflow(self, client, mock_transcription_service, sample_audio_file):
        """Test del flujo completo de transcripción"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    # 1. Detectar idioma
                    detect_response = client.post(
                        "/transcription/detect-language",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert detect_response.status_code == status.HTTP_200_OK
                    
                    # 2. Transcribir
                    transcribe_response = client.post(
                        "/transcription/transcribe",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert transcribe_response.status_code == status.HTTP_200_OK
                    
                    # 3. Resumir
                    text = transcribe_response.json()["text"]
                    summarize_response = client.post(
                        "/transcription/summarize",
                        json={"text": text}
                    )
                    assert summarize_response.status_code == status.HTTP_200_OK




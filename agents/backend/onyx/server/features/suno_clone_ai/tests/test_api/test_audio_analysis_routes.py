"""
Tests para las rutas de análisis de audio
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.audio_analysis import router
from services.audio_analysis import AudioAnalysisService


@pytest.fixture
def mock_audio_analysis_service():
    """Mock del servicio de análisis de audio"""
    service = Mock(spec=AudioAnalysisService)
    
    analysis_result = Mock()
    analysis_result.bpm = 120.0
    analysis_result.key = "C"
    analysis_result.tempo = "moderate"
    analysis_result.energy = 0.8
    analysis_result.danceability = 0.7
    analysis_result.valence = 0.6
    
    service.analyze_audio = Mock(return_value=analysis_result)
    service.detect_bpm = Mock(return_value=120.0)
    service.detect_key = Mock(return_value="C")
    service.analyze_spectrum = Mock(return_value={"frequencies": [440, 880]})
    
    return service


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    return io.BytesIO(b"fake audio content for analysis")


@pytest.fixture
def client(mock_audio_analysis_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.audio_analysis.get_audio_analysis_service', return_value=mock_audio_analysis_service):
        with patch('api.routes.audio_analysis.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeAudio:
    """Tests para análisis de audio"""
    
    def test_analyze_audio_success(self, client, mock_audio_analysis_service, sample_audio_file):
        """Test de análisis exitoso"""
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
                        "/audio-analysis/analyze",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bpm" in data
        assert "key" in data
        assert "energy" in data
    
    def test_detect_bpm_success(self, client, mock_audio_analysis_service, sample_audio_file):
        """Test de detección de BPM"""
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
                        "/audio-analysis/detect-bpm",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bpm" in data
    
    def test_detect_key_success(self, client, mock_audio_analysis_service, sample_audio_file):
        """Test de detección de tonalidad"""
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
                        "/audio-analysis/detect-key",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "key" in data


@pytest.mark.integration
@pytest.mark.api
class TestAudioAnalysisIntegration:
    """Tests de integración para análisis de audio"""
    
    def test_full_analysis_workflow(self, client, mock_audio_analysis_service, sample_audio_file):
        """Test del flujo completo de análisis"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    # 1. Análisis completo
                    analyze_response = client.post(
                        "/audio-analysis/analyze",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert analyze_response.status_code == status.HTTP_200_OK
                    
                    # 2. Detectar BPM
                    bpm_response = client.post(
                        "/audio-analysis/detect-bpm",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert bpm_response.status_code == status.HTTP_200_OK
                    
                    # 3. Detectar tonalidad
                    key_response = client.post(
                        "/audio-analysis/detect-key",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert key_response.status_code == status.HTTP_200_OK




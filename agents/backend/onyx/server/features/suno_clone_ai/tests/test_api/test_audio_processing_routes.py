"""
Tests para las rutas de procesamiento de audio
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.audio_processing import router
from services.audio_processor import AudioProcessor


@pytest.fixture
def mock_audio_processor():
    """Mock del procesador de audio"""
    processor = Mock(spec=AudioProcessor)
    
    processed_result = Mock()
    processed_result.output_path = "/tmp/processed.wav"
    processed_result.duration = 180.0
    
    processor.edit_audio = Mock(return_value=processed_result)
    processor.mix_audio = Mock(return_value=processed_result)
    processor.normalize_audio = Mock(return_value=processed_result)
    
    return processor


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    return io.BytesIO(b"fake audio content for processing")


@pytest.fixture
def client(mock_audio_processor):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.audio_processing.get_audio_processor', return_value=mock_audio_processor):
        with patch('api.routes.audio_processing.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestEditAudio:
    """Tests para edición de audio"""
    
    def test_edit_audio_success(self, client, mock_audio_processor, sample_audio_file):
        """Test de edición exitosa"""
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
                        "/audio-processing/edit",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                        data={"operation": "trim", "start": 0, "end": 30}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "output_path" in data or "message" in data


@pytest.mark.unit
@pytest.mark.api
class TestMixAudio:
    """Tests para mezcla de audio"""
    
    def test_mix_audio_success(self, client, mock_audio_processor, sample_audio_file):
        """Test de mezcla exitosa"""
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
                        "/audio-processing/mix",
                        files={
                            "file1": ("test1.wav", sample_audio_file, "audio/wav"),
                            "file2": ("test2.wav", sample_audio_file, "audio/wav")
                        }
                    )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestNormalizeAudio:
    """Tests para normalización de audio"""
    
    def test_normalize_audio_success(self, client, mock_audio_processor, sample_audio_file):
        """Test de normalización exitosa"""
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
                        "/audio-processing/normalize",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.integration
@pytest.mark.api
class TestAudioProcessingIntegration:
    """Tests de integración para procesamiento de audio"""
    
    def test_full_processing_workflow(self, client, mock_audio_processor, sample_audio_file):
        """Test del flujo completo de procesamiento"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    # 1. Editar audio
                    edit_response = client.post(
                        "/audio-processing/edit",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                        data={"operation": "trim"}
                    )
                    assert edit_response.status_code == status.HTTP_200_OK
                    
                    # 2. Normalizar audio
                    normalize_response = client.post(
                        "/audio-processing/normalize",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert normalize_response.status_code == status.HTTP_200_OK

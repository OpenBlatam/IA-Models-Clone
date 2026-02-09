"""
Tests para las rutas de karaoke
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.karaoke import router
from services.karaoke import KaraokeService
from services.lyrics_sync import LyricsSynchronizer


@pytest.fixture
def mock_karaoke_service():
    """Mock del servicio de karaoke"""
    service = Mock(spec=KaraokeService)
    service.create_karaoke_track = Mock(return_value={
        "success": True,
        "output_path": "/tmp/karaoke.wav",
        "method": "center"
    })
    return service


@pytest.fixture
def mock_lyrics_synchronizer():
    """Mock del sincronizador de letras"""
    synchronizer = Mock(spec=LyricsSynchronizer)
    synchronizer.sync_lyrics = Mock(return_value={
        "success": True,
        "timestamps": []
    })
    return synchronizer


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    return io.BytesIO(b"fake audio content for karaoke testing")


@pytest.fixture
def client(mock_karaoke_service, mock_lyrics_synchronizer):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.karaoke.get_karaoke_service', return_value=mock_karaoke_service):
        with patch('api.routes.karaoke.get_lyrics_synchronizer', return_value=mock_lyrics_synchronizer):
            with patch('api.routes.karaoke.get_current_user', return_value={"user_id": "test_user"}):
                yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateKaraokeTrack:
    """Tests para el endpoint de creación de pista de karaoke"""
    
    def test_create_karaoke_track_success(self, client, mock_karaoke_service, sample_audio_file):
        """Test de creación exitosa de pista de karaoke"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.exists', return_value=True):
                with patch('os.unlink'):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = Mock()
                        mock_file.name = "/tmp/input.wav"
                        mock_file.__enter__ = Mock(return_value=mock_file)
                        mock_file.__exit__ = Mock(return_value=None)
                        mock_file.write = Mock()
                        mock_temp.return_value = mock_file
                        
                        mock_output = Mock()
                        mock_output.read = Mock(return_value=b"karaoke audio content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/karaoke/create-track",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={"method": "center"}
                        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert "karaoke" in response.headers.get("content-disposition", "").lower()
        
        # Verificar que se llamó al servicio
        assert mock_karaoke_service.create_karaoke_track.called
    
    def test_create_karaoke_track_default_method(self, client, mock_karaoke_service, sample_audio_file):
        """Test con método por defecto"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.exists', return_value=True):
                with patch('os.unlink'):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = Mock()
                        mock_file.name = "/tmp/input.wav"
                        mock_file.__enter__ = Mock(return_value=mock_file)
                        mock_file.__exit__ = Mock(return_value=None)
                        mock_file.write = Mock()
                        mock_temp.return_value = mock_file
                        
                        mock_output = Mock()
                        mock_output.read = Mock(return_value=b"karaoke content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/karaoke/create-track",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_karaoke_service.create_karaoke_track.call_args
        assert call_args[0][2] == "center"  # Método por defecto
    
    def test_create_karaoke_track_different_methods(self, client, mock_karaoke_service, sample_audio_file):
        """Test con diferentes métodos de eliminación de voces"""
        methods = ["center", "stereo", "ml", "spectral"]
        
        for method in methods:
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('os.path.exists', return_value=True):
                    with patch('os.unlink'):
                        with patch('builtins.open', create=True) as mock_open:
                            mock_file = Mock()
                            mock_file.name = "/tmp/input.wav"
                            mock_file.__enter__ = Mock(return_value=mock_file)
                            mock_file.__exit__ = Mock(return_value=None)
                            mock_file.write = Mock()
                            mock_temp.return_value = mock_file
                            
                            mock_output = Mock()
                            mock_output.read = Mock(return_value=b"content")
                            mock_output.__enter__ = Mock(return_value=mock_output)
                            mock_output.__exit__ = Mock(return_value=None)
                            mock_open.return_value = mock_output
                            
                            response = client.post(
                                "/karaoke/create-track",
                                files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                                data={"method": method}
                            )
                            
                            assert response.status_code == status.HTTP_200_OK
                            call_args = mock_karaoke_service.create_karaoke_track.call_args
                            assert call_args[0][2] == method
    
    def test_create_karaoke_track_error_handling(self, client, mock_karaoke_service, sample_audio_file):
        """Test de manejo de errores"""
        mock_karaoke_service.create_karaoke_track.side_effect = Exception("Karaoke creation failed")
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.exists', return_value=True):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/input.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/karaoke/create-track",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error creating karaoke track" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestEvaluatePerformance:
    """Tests para el endpoint de evaluación de rendimiento"""
    
    def test_evaluate_performance_success(self, client, mock_karaoke_service, sample_audio_file):
        """Test de evaluación exitosa"""
        mock_karaoke_service.evaluate_performance = Mock(return_value={
            "score": 85.5,
            "pitch_accuracy": 0.92,
            "rhythm_accuracy": 0.88,
            "timing_score": 0.90,
            "feedback": "Good performance!"
        })
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.exists', return_value=True):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/input.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/karaoke/evaluate",
                        files={
                            "original": ("original.wav", sample_audio_file, "audio/wav"),
                            "performance": ("performance.wav", sample_audio_file, "audio/wav")
                        }
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "score" in data
        assert "pitch_accuracy" in data
        assert "rhythm_accuracy" in data
        assert data["score"] == 85.5
    
    def test_evaluate_performance_missing_files(self, client):
        """Test con archivos faltantes"""
        response = client.post(
            "/karaoke/evaluate",
            files={"original": ("original.wav", io.BytesIO(b"audio"), "audio/wav")}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_evaluate_performance_error_handling(self, client, mock_karaoke_service, sample_audio_file):
        """Test de manejo de errores en evaluación"""
        mock_karaoke_service.evaluate_performance = Mock(side_effect=Exception("Evaluation failed"))
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.exists', return_value=True):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/input.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/karaoke/evaluate",
                        files={
                            "original": ("original.wav", sample_audio_file, "audio/wav"),
                            "performance": ("performance.wav", sample_audio_file, "audio/wav")
                        }
                    )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error evaluating performance" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.api
class TestKaraokeIntegration:
    """Tests de integración para karaoke"""
    
    def test_full_karaoke_workflow(self, client, mock_karaoke_service, sample_audio_file):
        """Test del flujo completo de karaoke"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.exists', return_value=True):
                with patch('os.unlink'):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = Mock()
                        mock_file.name = "/tmp/input.wav"
                        mock_file.__enter__ = Mock(return_value=mock_file)
                        mock_file.__exit__ = Mock(return_value=None)
                        mock_file.write = Mock()
                        mock_temp.return_value = mock_file
                        
                        mock_output = Mock()
                        mock_output.read = Mock(return_value=b"karaoke audio")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        # Crear pista de karaoke
                        response = client.post(
                            "/karaoke/create-track",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={"method": "ml"}
                        )
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.content) > 0
        assert mock_karaoke_service.create_karaoke_track.called




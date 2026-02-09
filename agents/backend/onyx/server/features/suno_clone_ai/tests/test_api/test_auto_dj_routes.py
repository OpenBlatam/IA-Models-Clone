"""
Tests para las rutas de DJ automático
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.auto_dj import router
from services.auto_dj import AutoDJService


@pytest.fixture
def mock_auto_dj_service():
    """Mock del servicio de DJ automático"""
    service = Mock(spec=AutoDJService)
    
    # Mock de track info
    track_info = Mock()
    track_info.track_id = "track-123"
    track_info.bpm = 120.0
    track_info.key = "C"
    track_info.energy = 0.8
    track_info.duration = 180.0
    
    service.analyze_track = Mock(return_value=track_info)
    
    # Mock de DJ set
    dj_set = Mock()
    dj_set.set_id = "set-456"
    dj_set.duration = 3600.0
    dj_set.track_count = 10
    
    service.create_mix = Mock(return_value=dj_set)
    service.generate_playlist = Mock(return_value=["song-1", "song-2", "song-3"])
    
    return service


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    return io.BytesIO(b"fake audio content for DJ analysis")


@pytest.fixture
def client(mock_auto_dj_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.auto_dj.get_auto_dj_service', return_value=mock_auto_dj_service):
        with patch('api.routes.auto_dj.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeTrack:
    """Tests para análisis de pista"""
    
    def test_analyze_track_success(self, client, mock_auto_dj_service, sample_audio_file):
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
                        "/auto-dj/analyze",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "track_id" in data
        assert "bpm" in data
        assert data["bpm"] == 120.0
        assert "key" in data
        assert "energy" in data
        assert "duration" in data
    
    def test_analyze_track_error_handling(self, client, mock_auto_dj_service, sample_audio_file):
        """Test de manejo de errores"""
        mock_auto_dj_service.analyze_track.side_effect = Exception("Analysis failed")
        
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
                        "/auto-dj/analyze",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error analyzing track" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestCreateMix:
    """Tests para crear mix"""
    
    def test_create_mix_success(self, client, mock_auto_dj_service):
        """Test de creación exitosa de mix"""
        response = client.post(
            "/auto-dj/create-mix",
            json={
                "track_paths": ["/path/to/track1.wav", "/path/to/track2.wav"],
                "transition_type": "crossfade",
                "transition_duration": 2.0
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "set_id" in data
        assert "duration" in data
    
    def test_create_mix_different_transitions(self, client, mock_auto_dj_service):
        """Test con diferentes tipos de transición"""
        transitions = ["crossfade", "cut", "fade"]
        
        for transition in transitions:
            response = client.post(
                "/auto-dj/create-mix",
                json={
                    "track_paths": ["/path/to/track1.wav"],
                    "transition_type": transition
                }
            )
            assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestGeneratePlaylist:
    """Tests para generar playlist"""
    
    def test_generate_playlist_success(self, client, mock_auto_dj_service):
        """Test de generación exitosa de playlist"""
        response = client.post(
            "/auto-dj/generate-playlist",
            json={
                "seed_tracks": ["track-1", "track-2"],
                "duration_minutes": 60
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "playlist" in data or "tracks" in data


@pytest.mark.integration
@pytest.mark.api
class TestAutoDJIntegration:
    """Tests de integración para DJ automático"""
    
    def test_full_dj_workflow(self, client, mock_auto_dj_service, sample_audio_file):
        """Test del flujo completo de DJ automático"""
        # 1. Analizar pista
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    analyze_response = client.post(
                        "/auto-dj/analyze",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert analyze_response.status_code == status.HTTP_200_OK
        
        # 2. Crear mix
        mix_response = client.post(
            "/auto-dj/create-mix",
            json={
                "track_paths": ["/path/to/track1.wav", "/path/to/track2.wav"],
                "transition_type": "crossfade"
            }
        )
        assert mix_response.status_code == status.HTTP_200_OK
        
        # 3. Generar playlist
        playlist_response = client.post(
            "/auto-dj/generate-playlist",
            json={"seed_tracks": ["track-1"], "duration_minutes": 30}
        )
        assert playlist_response.status_code == status.HTTP_200_OK




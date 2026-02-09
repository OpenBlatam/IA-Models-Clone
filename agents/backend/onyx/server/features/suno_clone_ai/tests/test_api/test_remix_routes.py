"""
Tests para las rutas de remix y mashup
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
import io
import os
import tempfile

from api.routes.remix import router
from services.audio_remix import AudioRemixer, RemixConfig


@pytest.fixture
def mock_audio_remixer():
    """Mock del servicio de remix"""
    remixer = Mock(spec=AudioRemixer)
    remixer.remix = Mock(return_value={"success": True, "output_path": "/tmp/remix.wav"})
    remixer.mashup = Mock(return_value={"success": True, "output_path": "/tmp/mashup.wav"})
    return remixer


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    audio_content = b"fake audio content for testing"
    return io.BytesIO(audio_content)


@pytest.fixture
def client(mock_audio_remixer):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.remix.get_audio_remixer', return_value=mock_audio_remixer):
        with patch('api.routes.remix.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateRemix:
    """Tests para el endpoint de creación de remix"""
    
    def test_create_remix_success(self, client, mock_audio_remixer, sample_audio_file):
        """Test de creación exitosa de remix"""
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
                        mock_output.read = Mock(return_value=b"remix audio content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/create",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={
                                "target_bpm": 120.0,
                                "fade_in": 2.0,
                                "fade_out": 2.0,
                                "volume": 0.8
                            }
                        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert "remix" in response.headers.get("content-disposition", "").lower()
        
        # Verificar que se llamó al remixer
        assert mock_audio_remixer.remix.called
    
    def test_create_remix_default_values(self, client, mock_audio_remixer, sample_audio_file):
        """Test con valores por defecto"""
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
                        mock_output.read = Mock(return_value=b"remix content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/create",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verificar que se llamó con valores por defecto
        call_args = mock_audio_remixer.remix.call_args
        config = call_args[0][2]  # RemixConfig es el tercer argumento
        assert config.fade_in == 0.0
        assert config.fade_out == 0.0
        assert config.volume == 1.0
    
    def test_create_remix_with_bpm(self, client, mock_audio_remixer, sample_audio_file):
        """Test con BPM específico"""
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
                        mock_output.read = Mock(return_value=b"remix content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/create",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={"target_bpm": 140.0}
                        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_audio_remixer.remix.call_args
        config = call_args[0][2]
        assert config.target_bpm == 140.0
    
    def test_create_remix_error_handling(self, client, mock_audio_remixer, sample_audio_file):
        """Test de manejo de errores"""
        mock_audio_remixer.remix.side_effect = Exception("Remix failed")
        
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
                        "/remix/create",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error creating remix" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestCreateMashup:
    """Tests para el endpoint de creación de mashup"""
    
    def test_create_mashup_success(self, client, mock_audio_remixer, sample_audio_file):
        """Test de creación exitosa de mashup"""
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
                        mock_output.read = Mock(return_value=b"mashup audio content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/mashup",
                            files=[
                                ("files", ("song1.wav", sample_audio_file, "audio/wav")),
                                ("files", ("song2.wav", sample_audio_file, "audio/wav"))
                            ],
                            data={
                                "target_bpm": 120.0,
                                "crossfade": 1.0,
                                "fade_in": 2.0,
                                "fade_out": 2.0,
                                "volume": 0.9
                            }
                        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert "mashup" in response.headers.get("content-disposition", "").lower()
        
        # Verificar que se llamó al remixer con múltiples archivos
        assert mock_audio_remixer.mashup.called
        call_args = mock_audio_remixer.mashup.call_args
        assert len(call_args[0][0]) == 2  # Dos archivos de entrada
    
    def test_create_mashup_single_file(self, client, mock_audio_remixer, sample_audio_file):
        """Test de mashup con un solo archivo"""
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
                        mock_output.read = Mock(return_value=b"mashup content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/mashup",
                            files=[("files", ("song1.wav", sample_audio_file, "audio/wav"))]
                        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_audio_remixer.mashup.call_args
        assert len(call_args[0][0]) == 1
    
    def test_create_mashup_multiple_files(self, client, mock_audio_remixer, sample_audio_file):
        """Test de mashup con múltiples archivos"""
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
                        mock_output.read = Mock(return_value=b"mashup content")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/mashup",
                            files=[
                                ("files", ("song1.wav", sample_audio_file, "audio/wav")),
                                ("files", ("song2.wav", sample_audio_file, "audio/wav")),
                                ("files", ("song3.wav", sample_audio_file, "audio/wav"))
                            ]
                        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_audio_remixer.mashup.call_args
        assert len(call_args[0][0]) == 3
    
    def test_create_mashup_error_handling(self, client, mock_audio_remixer, sample_audio_file):
        """Test de manejo de errores en mashup"""
        mock_audio_remixer.mashup.side_effect = Exception("Mashup failed")
        
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
                        "/remix/mashup",
                        files=[("files", ("song1.wav", sample_audio_file, "audio/wav"))]
                    )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error creating mashup" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestRemixValidation:
    """Tests de validación de parámetros de remix"""
    
    def test_remix_volume_range(self, client, mock_audio_remixer, sample_audio_file):
        """Test de validación de rango de volumen"""
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
                        
                        # Volumen válido
                        response = client.post(
                            "/remix/create",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={"volume": 0.5}
                        )
                        assert response.status_code == status.HTTP_200_OK
    
    def test_remix_negative_fade(self, client, mock_audio_remixer, sample_audio_file):
        """Test con fade negativo (debería permitirse o validarse)"""
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
                            "/remix/create",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={"fade_in": -1.0}
                        )
                        # Depende de la validación del servicio
                        assert response.status_code in [
                            status.HTTP_200_OK,
                            status.HTTP_422_UNPROCESSABLE_ENTITY
                        ]


@pytest.mark.integration
@pytest.mark.api
class TestRemixIntegration:
    """Tests de integración para remix"""
    
    def test_full_remix_workflow(self, client, mock_audio_remixer, sample_audio_file):
        """Test del flujo completo de remix"""
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
                        mock_output.read = Mock(return_value=b"remix audio")
                        mock_output.__enter__ = Mock(return_value=mock_output)
                        mock_output.__exit__ = Mock(return_value=None)
                        mock_open.return_value = mock_output
                        
                        response = client.post(
                            "/remix/create",
                            files={"file": ("test.wav", sample_audio_file, "audio/wav")},
                            data={
                                "target_bpm": 128.0,
                                "fade_in": 1.5,
                                "fade_out": 1.5,
                                "volume": 0.85
                            }
                        )
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.content) > 0
        assert mock_audio_remixer.remix.called




"""
Tests mejorados para las rutas de streaming
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.streaming import router
from services.audio_streaming import AudioStreamer, StreamConfig


@pytest.fixture
def mock_audio_streamer():
    """Mock del servicio de streaming"""
    streamer = AsyncMock(spec=AudioStreamer)
    streamer.create_stream = AsyncMock(return_value={
        "stream_id": "test-stream-123",
        "status": "active",
        "audio_path": "/tmp/audio.wav"
    })
    
    async def mock_stream_chunks(stream_id):
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        for chunk in chunks:
            yield chunk
    
    streamer.stream_chunks = mock_stream_chunks
    streamer.pause_stream = AsyncMock(return_value=True)
    streamer.resume_stream = AsyncMock(return_value=True)
    streamer.stop_stream = AsyncMock(return_value=True)
    streamer.seek_stream = AsyncMock(return_value=True)
    streamer.get_stream_stats = AsyncMock(return_value={
        "bytes_streamed": 1024,
        "chunks_sent": 10,
        "status": "active"
    })
    return streamer


@pytest.fixture
def client(mock_audio_streamer):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.streaming.get_audio_streamer', return_value=mock_audio_streamer):
        with patch('api.routes.streaming.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateStream:
    """Tests para el endpoint de creación de stream"""
    
    def test_create_stream_success(self, client, mock_audio_streamer):
        """Test de creación exitosa de stream"""
        response = client.post(
            "/streaming/create",
            json={
                "audio_path": "/tmp/test.wav",
                "sample_rate": 44100,
                "channels": 2,
                "format": "wav"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "stream_id" in data
        assert data["status"] == "active"
        
        mock_audio_streamer.create_stream.assert_called_once()
    
    def test_create_stream_default_values(self, client, mock_audio_streamer):
        """Test con valores por defecto"""
        response = client.post(
            "/streaming/create",
            json={"audio_path": "/tmp/test.wav"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_audio_streamer.create_stream.call_args
        config = call_args[0][2]  # StreamConfig
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.format == "wav"
    
    def test_create_stream_different_formats(self, client, mock_audio_streamer):
        """Test con diferentes formatos"""
        formats = ["wav", "mp3", "ogg", "flac"]
        
        for fmt in formats:
            response = client.post(
                "/streaming/create",
                json={
                    "audio_path": f"/tmp/test.{fmt}",
                    "format": fmt
                }
            )
            assert response.status_code == status.HTTP_200_OK
    
    def test_create_stream_error_handling(self, client, mock_audio_streamer):
        """Test de manejo de errores"""
        mock_audio_streamer.create_stream.side_effect = Exception("Stream creation failed")
        
        response = client.post(
            "/streaming/create",
            json={"audio_path": "/tmp/test.wav"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error creating stream" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestStreamAudio:
    """Tests para el endpoint de streaming de audio"""
    
    @pytest.mark.asyncio
    async def test_stream_audio_success(self, client, mock_audio_streamer):
        """Test de streaming exitoso"""
        response = client.get("/streaming/stream/test-stream-123")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert "stream" in response.headers.get("content-disposition", "").lower()
    
    def test_stream_audio_not_found(self, client, mock_audio_streamer):
        """Test cuando el stream no existe"""
        async def mock_stream_chunks_not_found(stream_id):
            raise ValueError("Stream not found")
        
        mock_audio_streamer.stream_chunks = mock_stream_chunks_not_found
        
        response = client.get("/streaming/stream/nonexistent")
        
        # Puede ser 200 con error en el stream o 500
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]


@pytest.mark.unit
@pytest.mark.api
class TestControlStream:
    """Tests para control de stream"""
    
    def test_pause_stream_success(self, client, mock_audio_streamer):
        """Test de pausar stream exitosamente"""
        response = client.post(
            "/streaming/test-stream-123/pause"
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_audio_streamer.pause_stream.assert_called_once_with("test-stream-123")
    
    def test_resume_stream_success(self, client, mock_audio_streamer):
        """Test de reanudar stream exitosamente"""
        response = client.post(
            "/streaming/test-stream-123/resume"
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_audio_streamer.resume_stream.assert_called_once_with("test-stream-123")
    
    def test_stop_stream_success(self, client, mock_audio_streamer):
        """Test de detener stream exitosamente"""
        response = client.post(
            "/streaming/test-stream-123/stop"
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_audio_streamer.stop_stream.assert_called_once_with("test-stream-123")
    
    def test_seek_stream_success(self, client, mock_audio_streamer):
        """Test de seek en stream exitosamente"""
        response = client.post(
            "/streaming/test-stream-123/seek",
            json={"position": 30.0}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_audio_streamer.seek_stream.assert_called_once_with("test-stream-123", 30.0)
    
    def test_seek_stream_invalid_position(self, client):
        """Test con posición inválida"""
        response = client.post(
            "/streaming/test-stream-123/seek",
            json={"position": -1.0}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.unit
@pytest.mark.api
class TestGetStreamStats:
    """Tests para obtener estadísticas de stream"""
    
    def test_get_stream_stats_success(self, client, mock_audio_streamer):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/streaming/test-stream-123/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bytes_streamed" in data
        assert "chunks_sent" in data
        assert data["bytes_streamed"] == 1024
        assert data["chunks_sent"] == 10
    
    def test_get_stream_stats_not_found(self, client, mock_audio_streamer):
        """Test cuando el stream no existe"""
        mock_audio_streamer.get_stream_stats.side_effect = ValueError("Stream not found")
        
        response = client.get("/streaming/nonexistent/stats")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.integration
@pytest.mark.api
class TestStreamingIntegration:
    """Tests de integración para streaming"""
    
    def test_full_streaming_workflow(self, client, mock_audio_streamer):
        """Test del flujo completo de streaming"""
        # 1. Crear stream
        create_response = client.post(
            "/streaming/create",
            json={"audio_path": "/tmp/test.wav"}
        )
        assert create_response.status_code == status.HTTP_200_OK
        stream_id = create_response.json()["stream_id"]
        
        # 2. Obtener estadísticas
        stats_response = client.get(f"/streaming/{stream_id}/stats")
        assert stats_response.status_code == status.HTTP_200_OK
        
        # 3. Pausar
        pause_response = client.post(f"/streaming/{stream_id}/pause")
        assert pause_response.status_code == status.HTTP_200_OK
        
        # 4. Reanudar
        resume_response = client.post(f"/streaming/{stream_id}/resume")
        assert resume_response.status_code == status.HTTP_200_OK
        
        # 5. Seek
        seek_response = client.post(
            f"/streaming/{stream_id}/seek",
            json={"position": 15.0}
        )
        assert seek_response.status_code == status.HTTP_200_OK
        
        # 6. Detener
        stop_response = client.post(f"/streaming/{stream_id}/stop")
        assert stop_response.status_code == status.HTTP_200_OK




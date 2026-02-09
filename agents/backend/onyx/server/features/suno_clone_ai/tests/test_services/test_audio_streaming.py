"""
Tests para audio streaming service
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from services.audio_streaming import (
    AudioStreamer,
    StreamConfig,
    StreamStats
)


@pytest.fixture
def audio_streamer():
    """Fixture para AudioStreamer"""
    return AudioStreamer()


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    # Crear archivo temporal con datos de audio simulados
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        # Escribir datos simulados (en producción sería un archivo WAV real)
        f.write(b"fake audio data for testing")
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
@pytest.mark.services
class TestStreamConfig:
    """Tests para StreamConfig"""
    
    def test_stream_config_default(self):
        """Test de configuración por defecto"""
        config = StreamConfig()
        
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.format == "wav"
        assert config.bitrate == 192
        assert config.buffer_size == 4096
        assert config.adaptive_quality is True
    
    def test_stream_config_custom(self):
        """Test de configuración personalizada"""
        config = StreamConfig(
            sample_rate=48000,
            channels=1,
            format="mp3",
            bitrate=256,
            buffer_size=8192,
            adaptive_quality=False
        )
        
        assert config.sample_rate == 48000
        assert config.channels == 1
        assert config.format == "mp3"
        assert config.bitrate == 256
        assert config.buffer_size == 8192
        assert config.adaptive_quality is False


@pytest.mark.unit
@pytest.mark.services
class TestStreamStats:
    """Tests para StreamStats"""
    
    def test_stream_stats_default(self):
        """Test de estadísticas por defecto"""
        stats = StreamStats()
        
        assert stats.bytes_sent == 0
        assert stats.chunks_sent == 0
        assert stats.current_bitrate == 0.0
        assert stats.buffer_level == 0.0
        assert stats.dropped_chunks == 0
        assert stats.start_time is not None


@pytest.mark.unit
@pytest.mark.services
class TestAudioStreamer:
    """Tests para AudioStreamer"""
    
    def test_audio_streamer_init(self, audio_streamer):
        """Test de inicialización"""
        assert audio_streamer.active_streams == {}
        assert audio_streamer.stream_stats == {}
    
    @pytest.mark.asyncio
    @patch('services.audio_streaming.sf')
    async def test_create_stream_success(self, mock_sf, audio_streamer, sample_audio_file):
        """Test de creación exitosa de stream"""
        # Mock de soundfile
        mock_sf.read.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        
        config = StreamConfig(sample_rate=44100, channels=1)
        
        stream_info = await audio_streamer.create_stream(
            "stream-123",
            sample_audio_file,
            config=config
        )
        
        assert stream_info is not None
        assert "stream_id" in stream_info
        assert stream_info["stream_id"] == "stream-123"
    
    @pytest.mark.asyncio
    async def test_create_stream_no_audio_libs(self, audio_streamer, sample_audio_file):
        """Test cuando no hay librerías de audio"""
        with patch('services.audio_streaming.AUDIO_LIBS_AVAILABLE', False):
            with pytest.raises(Exception, match="Audio libraries not available"):
                await audio_streamer.create_stream("stream-123", sample_audio_file)
    
    @pytest.mark.asyncio
    @patch('services.audio_streaming.sf')
    async def test_get_stream_chunk(self, mock_sf, audio_streamer, sample_audio_file):
        """Test de obtención de chunk de stream"""
        mock_sf.read.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        
        config = StreamConfig(buffer_size=1024)
        await audio_streamer.create_stream("stream-123", sample_audio_file, config=config)
        
        chunk = await audio_streamer.get_stream_chunk("stream-123")
        
        assert chunk is not None
        assert "data" in chunk or "chunk" in chunk
    
    @pytest.mark.asyncio
    async def test_get_stream_chunk_not_found(self, audio_streamer):
        """Test de chunk de stream no encontrado"""
        with pytest.raises(ValueError, match="not found"):
            await audio_streamer.get_stream_chunk("nonexistent")
    
    @pytest.mark.asyncio
    @patch('services.audio_streaming.sf')
    async def test_stop_stream(self, mock_sf, audio_streamer, sample_audio_file):
        """Test de detener stream"""
        mock_sf.read.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        
        await audio_streamer.create_stream("stream-123", sample_audio_file)
        await audio_streamer.stop_stream("stream-123")
        
        assert "stream-123" not in audio_streamer.active_streams
    
    @pytest.mark.asyncio
    @patch('services.audio_streaming.sf')
    async def test_get_stream_stats(self, mock_sf, audio_streamer, sample_audio_file):
        """Test de obtención de estadísticas"""
        mock_sf.read.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        
        await audio_streamer.create_stream("stream-123", sample_audio_file)
        stats = audio_streamer.get_stream_stats("stream-123")
        
        assert stats is not None
        assert isinstance(stats, StreamStats)




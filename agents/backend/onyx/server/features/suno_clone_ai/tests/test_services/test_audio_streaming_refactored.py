"""
Tests refactorizados para audio streaming service
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from services.audio_streaming import (
    AudioStreamer,
    StreamConfig,
    StreamStats
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestStreamConfigRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para StreamConfig"""
    
    def test_stream_config_default(self):
        """Test de configuración por defecto"""
        config = StreamConfig()
        
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.format == "wav"
        assert config.bitrate == 192
        assert config.buffer_size == 4096
        assert config.adaptive_quality is True
    
    @pytest.mark.parametrize("sample_rate,channels,format,bitrate,buffer_size,adaptive", [
        (48000, 1, "mp3", 256, 8192, False),
        (44100, 2, "wav", 192, 4096, True),
        (96000, 2, "ogg", 320, 16384, True)
    ])
    def test_stream_config_custom(self, sample_rate, channels, format, bitrate, buffer_size, adaptive):
        """Test de configuración personalizada con diferentes valores"""
        config = StreamConfig(
            sample_rate=sample_rate,
            channels=channels,
            format=format,
            bitrate=bitrate,
            buffer_size=buffer_size,
            adaptive_quality=adaptive
        )
        
        assert config.sample_rate == sample_rate
        assert config.channels == channels
        assert config.format == format
        assert config.bitrate == bitrate
        assert config.buffer_size == buffer_size
        assert config.adaptive_quality == adaptive


class TestStreamStatsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para StreamStats"""
    
    def test_stream_stats_default(self):
        """Test de estadísticas por defecto"""
        stats = StreamStats()
        
        assert stats.bytes_sent == 0
        assert stats.chunks_sent == 0
        assert stats.current_bitrate == 0.0
        assert stats.buffer_level == 0.0
        assert stats.dropped_chunks == 0
        assert stats.start_time is not None


class TestAudioStreamerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para AudioStreamer"""
    
    @pytest.fixture
    def audio_streamer(self):
        """Fixture para AudioStreamer"""
        return AudioStreamer()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Archivo de audio de prueba"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            f.write(b"fake audio data for testing")
            yield temp_path
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_audio_streamer_init(self, audio_streamer):
        """Test de inicialización"""
        assert audio_streamer.active_streams == {}
        assert audio_streamer.stream_stats == {}
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("sample_rate,channels", [
        (44100, 1),
        (44100, 2),
        (48000, 2)
    ])
    @patch('services.audio_streaming.sf')
    async def test_create_stream_success(self, mock_sf, audio_streamer, sample_audio_file, sample_rate, channels):
        """Test de creación exitosa de stream con diferentes configuraciones"""
        mock_sf.read.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        
        config = StreamConfig(sample_rate=sample_rate, channels=channels)
        
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




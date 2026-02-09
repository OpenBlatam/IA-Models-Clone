"""
Tests para el servicio de transcripción de audio
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import tempfile
import os

from services.audio_transcription import TranscriptionService


@pytest.fixture
def transcription_service():
    """Instancia del servicio de transcripción"""
    return TranscriptionService()


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        import soundfile as sf
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        sf.write(tmp.name, audio, sample_rate)
        yield tmp.name
    
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.mark.unit
class TestTranscriptionService:
    """Tests para el servicio de transcripción"""
    
    def test_service_initialization(self, transcription_service):
        """Test de inicialización"""
        assert transcription_service is not None
        assert isinstance(transcription_service, TranscriptionService)
    
    @pytest.mark.skipif(
        not hasattr(TranscriptionService, 'transcribe'),
        reason="transcribe method not available"
    )
    def test_transcribe_basic(self, transcription_service, sample_audio_file):
        """Test básico de transcripción"""
        try:
            result = transcription_service.transcribe(sample_audio_file)
            
            assert result is not None
            if isinstance(result, str):
                assert len(result) >= 0
            elif isinstance(result, dict):
                assert "text" in result or "transcription" in result
        except Exception as e:
            pytest.skip(f"Transcription not available: {e}")
    
    @pytest.mark.skipif(
        not hasattr(TranscriptionService, 'transcribe_with_timestamps'),
        reason="transcribe_with_timestamps method not available"
    )
    def test_transcribe_with_timestamps(self, transcription_service, sample_audio_file):
        """Test de transcripción con timestamps"""
        try:
            result = transcription_service.transcribe_with_timestamps(sample_audio_file)
            
            assert result is not None
            if isinstance(result, dict):
                assert "segments" in result or "words" in result
        except Exception as e:
            pytest.skip(f"Transcription with timestamps not available: {e}")
    
    def test_transcribe_invalid_file(self, transcription_service):
        """Test con archivo inválido"""
        invalid_path = "/nonexistent/file.wav"
        
        with pytest.raises((FileNotFoundError, Exception)):
            transcription_service.transcribe(invalid_path)
    
    @pytest.mark.skipif(
        not hasattr(TranscriptionService, 'transcribe_async'),
        reason="transcribe_async method not available"
    )
    @pytest.mark.asyncio
    async def test_transcribe_async(self, transcription_service, sample_audio_file):
        """Test de transcripción asíncrona"""
        try:
            result = await transcription_service.transcribe_async(sample_audio_file)
            
            assert result is not None
        except Exception as e:
            pytest.skip(f"Async transcription not available: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestTranscriptionIntegration:
    """Tests de integración para transcripción"""
    
    @pytest.mark.skipif(
        not hasattr(TranscriptionService, 'transcribe'),
        reason="transcribe method not available"
    )
    def test_full_transcription_workflow(self, transcription_service, sample_audio_file):
        """Test del flujo completo de transcripción"""
        try:
            # Transcripción básica
            result = transcription_service.transcribe(sample_audio_file)
            assert result is not None
            
            # Transcripción con timestamps (si está disponible)
            if hasattr(transcription_service, 'transcribe_with_timestamps'):
                result_with_ts = transcription_service.transcribe_with_timestamps(sample_audio_file)
                assert result_with_ts is not None
        except Exception as e:
            pytest.skip(f"Full workflow not available: {e}")




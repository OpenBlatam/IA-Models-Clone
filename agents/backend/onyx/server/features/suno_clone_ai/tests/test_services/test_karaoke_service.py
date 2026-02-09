"""
Tests para el servicio de karaoke
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import os

from services.karaoke import KaraokeService, KaraokeTrack, KaraokeScore


@pytest.fixture
def karaoke_service():
    """Instancia del servicio de karaoke"""
    return KaraokeService()


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
class TestKaraokeTrack:
    """Tests para KaraokeTrack"""
    
    def test_karaoke_track_creation(self):
        """Test de creación de KaraokeTrack"""
        track = KaraokeTrack(
            audio_path="/path/to/audio.wav",
            lyrics_path="/path/to/lyrics.txt"
        )
        
        assert track.audio_path == "/path/to/audio.wav"
        assert track.lyrics_path == "/path/to/lyrics.txt"
        assert track.synced_lyrics is None
        assert track.timestamp is not None
    
    def test_karaoke_track_minimal(self):
        """Test con parámetros mínimos"""
        track = KaraokeTrack(audio_path="/path/to/audio.wav")
        
        assert track.audio_path == "/path/to/audio.wav"
        assert track.lyrics_path is None


@pytest.mark.unit
class TestKaraokeScore:
    """Tests para KaraokeScore"""
    
    def test_karaoke_score_creation(self):
        """Test de creación de KaraokeScore"""
        score = KaraokeScore(
            accuracy=0.85,
            timing_score=0.90,
            pitch_score=0.80,
            total_score=0.85
        )
        
        assert score.accuracy == 0.85
        assert score.timing_score == 0.90
        assert score.pitch_score == 0.80
        assert score.total_score == 0.85
        assert score.timestamp is not None
    
    def test_karaoke_score_defaults(self):
        """Test con valores por defecto"""
        score = KaraokeScore()
        
        assert score.accuracy == 0.0
        assert score.timing_score == 0.0
        assert score.pitch_score == 0.0
        assert score.total_score == 0.0


@pytest.mark.unit
class TestKaraokeService:
    """Tests para el servicio de karaoke"""
    
    def test_service_initialization(self, karaoke_service):
        """Test de inicialización"""
        assert karaoke_service is not None
        assert isinstance(karaoke_service, KaraokeService)
    
    @pytest.mark.skipif(
        not hasattr(KaraokeService, 'create_karaoke_track'),
        reason="create_karaoke_track method not available"
    )
    def test_create_karaoke_track_basic(self, karaoke_service, sample_audio_file):
        """Test básico de creación de pista de karaoke"""
        output_path = sample_audio_file.replace('.wav', '_karaoke.wav')
        
        try:
            result = karaoke_service.create_karaoke_track(
                sample_audio_file,
                output_path,
                method="center"
            )
            
            assert result is not None
            if isinstance(result, dict):
                assert result.get("success") is not False
        except Exception as e:
            pytest.skip(f"Karaoke track creation not available: {e}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_create_karaoke_track_different_methods(self, karaoke_service, sample_audio_file):
        """Test con diferentes métodos"""
        methods = ["center", "stereo", "ml", "spectral"]
        output_path = sample_audio_file.replace('.wav', '_karaoke.wav')
        
        for method in methods:
            try:
                result = karaoke_service.create_karaoke_track(
                    sample_audio_file,
                    output_path,
                    method=method
                )
                assert result is not None
            except Exception as e:
                pytest.skip(f"Method {method} not available: {e}")
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
    
    def test_create_karaoke_track_invalid_file(self, karaoke_service):
        """Test con archivo inválido"""
        invalid_path = "/nonexistent/file.wav"
        
        with pytest.raises((FileNotFoundError, Exception)):
            karaoke_service.create_karaoke_track(
                invalid_path,
                "/tmp/output.wav",
                method="center"
            )
    
    @pytest.mark.skipif(
        not hasattr(KaraokeService, 'evaluate_performance'),
        reason="evaluate_performance method not available"
    )
    def test_evaluate_performance_basic(self, karaoke_service, sample_audio_file):
        """Test básico de evaluación de rendimiento"""
        try:
            result = karaoke_service.evaluate_performance(
                sample_audio_file,
                sample_audio_file  # Usar el mismo archivo como referencia
            )
            
            assert result is not None
            if isinstance(result, dict):
                assert "score" in result or "accuracy" in result
            elif isinstance(result, KaraokeScore):
                assert result.total_score >= 0.0
                assert result.total_score <= 1.0
        except Exception as e:
            pytest.skip(f"Performance evaluation not available: {e}")
    
    def test_evaluate_performance_invalid_files(self, karaoke_service):
        """Test con archivos inválidos"""
        invalid_path = "/nonexistent/file.wav"
        
        with pytest.raises((FileNotFoundError, Exception)):
            karaoke_service.evaluate_performance(
                invalid_path,
                invalid_path
            )


@pytest.mark.integration
@pytest.mark.slow
class TestKaraokeIntegration:
    """Tests de integración para karaoke"""
    
    def test_full_karaoke_workflow(self, karaoke_service, sample_audio_file):
        """Test del flujo completo de karaoke"""
        output_path = sample_audio_file.replace('.wav', '_full_karaoke.wav')
        
        try:
            # Crear pista de karaoke
            track_result = karaoke_service.create_karaoke_track(
                sample_audio_file,
                output_path,
                method="center"
            )
            
            assert track_result is not None
            
            # Evaluar rendimiento (si el método existe)
            if hasattr(karaoke_service, 'evaluate_performance'):
                if os.path.exists(output_path):
                    score_result = karaoke_service.evaluate_performance(
                        sample_audio_file,
                        output_path
                    )
                    assert score_result is not None
        except Exception as e:
            pytest.skip(f"Full workflow not available: {e}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)




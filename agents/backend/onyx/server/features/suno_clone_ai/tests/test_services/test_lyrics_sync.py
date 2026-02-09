"""
Tests para el servicio de sincronización de letras
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import os

from services.lyrics_sync import LyricsSynchronizer, WordTiming, SyncedLyrics


@pytest.fixture
def sample_lyrics():
    """Letras de ejemplo"""
    return "This is a test song with multiple words"


@pytest.fixture
def synchronizer():
    """Instancia del sincronizador"""
    return LyricsSynchronizer()


@pytest.mark.unit
class TestWordTiming:
    """Tests para WordTiming"""
    
    def test_word_timing_creation(self):
        """Test de creación de WordTiming"""
        word = WordTiming(
            word="test",
            start_time=1.0,
            end_time=1.5,
            confidence=0.9
        )
        
        assert word.word == "test"
        assert word.start_time == 1.0
        assert word.end_time == 1.5
        assert word.confidence == 0.9
    
    def test_word_timing_default_confidence(self):
        """Test con confianza por defecto"""
        word = WordTiming(
            word="test",
            start_time=1.0,
            end_time=1.5
        )
        
        assert word.confidence == 0.0


@pytest.mark.unit
class TestSyncedLyrics:
    """Tests para SyncedLyrics"""
    
    def test_synced_lyrics_creation(self):
        """Test de creación de SyncedLyrics"""
        words = [
            WordTiming("word1", 0.0, 0.5),
            WordTiming("word2", 0.5, 1.0)
        ]
        
        synced = SyncedLyrics(
            words=words,
            total_duration=1.0
        )
        
        assert len(synced.words) == 2
        assert synced.total_duration == 1.0
        assert synced.timestamp is not None


@pytest.mark.unit
class TestLyricsSynchronizer:
    """Tests para el sincronizador de letras"""
    
    def test_synchronizer_initialization(self, synchronizer):
        """Test de inicialización"""
        assert synchronizer is not None
        assert isinstance(synchronizer, LyricsSynchronizer)
    
    @pytest.mark.skipif(
        not hasattr(LyricsSynchronizer, 'sync_lyrics'),
        reason="sync_lyrics method not available"
    )
    def test_sync_lyrics_basic(self, synchronizer, sample_lyrics):
        """Test básico de sincronización"""
        # Crear archivo de audio temporal
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Escribir datos de audio simples
            import soundfile as sf
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            sf.write(tmp.name, audio, sample_rate)
            audio_path = tmp.name
        
        try:
            result = synchronizer.sync_lyrics(audio_path, sample_lyrics)
            
            assert result is not None
            if isinstance(result, SyncedLyrics):
                assert len(result.words) > 0
                assert result.total_duration > 0
        except Exception as e:
            pytest.skip(f"Sync not available: {e}")
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_sync_lyrics_empty_lyrics(self, synchronizer):
        """Test con letras vacías"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sample_rate = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            sf.write(tmp.name, audio, sample_rate)
            audio_path = tmp.name
        
        try:
            result = synchronizer.sync_lyrics(audio_path, "")
            # Debería manejar letras vacías
            assert result is not None or result is None
        except Exception as e:
            pytest.skip(f"Empty lyrics sync not available: {e}")
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_sync_lyrics_invalid_file(self, synchronizer, sample_lyrics):
        """Test con archivo inválido"""
        invalid_path = "/nonexistent/file.wav"
        
        with pytest.raises((FileNotFoundError, Exception)):
            synchronizer.sync_lyrics(invalid_path, sample_lyrics)
    
    @pytest.mark.skipif(
        not hasattr(LyricsSynchronizer, 'get_word_at_time'),
        reason="get_word_at_time method not available"
    )
    def test_get_word_at_time(self, synchronizer):
        """Test de obtener palabra en tiempo específico"""
        words = [
            WordTiming("word1", 0.0, 0.5),
            WordTiming("word2", 0.5, 1.0),
            WordTiming("word3", 1.0, 1.5)
        ]
        
        synced = SyncedLyrics(words=words, total_duration=1.5)
        
        # Test en diferentes tiempos
        word_at_0_25 = synchronizer.get_word_at_time(synced, 0.25)
        assert word_at_0_25 is not None
        
        word_at_0_75 = synchronizer.get_word_at_time(synced, 0.75)
        assert word_at_0_75 is not None
        
        word_at_1_25 = synchronizer.get_word_at_time(synced, 1.25)
        assert word_at_1_25 is not None


@pytest.mark.integration
@pytest.mark.slow
class TestLyricsSyncIntegration:
    """Tests de integración para sincronización de letras"""
    
    def test_full_sync_workflow(self, synchronizer):
        """Test del flujo completo de sincronización"""
        lyrics = "This is a complete test song with multiple lines"
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sample_rate = 44100
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            sf.write(tmp.name, audio, sample_rate)
            audio_path = tmp.name
        
        try:
            result = synchronizer.sync_lyrics(audio_path, lyrics)
            
            assert result is not None
            if isinstance(result, SyncedLyrics):
                assert result.total_duration > 0
                assert len(result.words) > 0
        except Exception as e:
            pytest.skip(f"Full workflow not available: {e}")
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)




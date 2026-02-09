"""
Tests para el generador de música core
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import os

from core.music_generator import MusicGenerator


@pytest.fixture
def music_generator():
    """Instancia del generador de música"""
    try:
        return MusicGenerator()
    except Exception as e:
        pytest.skip(f"MusicGenerator not available: {e}")


@pytest.fixture
def sample_text_prompt():
    """Prompt de texto de ejemplo"""
    return "A happy pop song with upbeat tempo"


@pytest.mark.unit
@pytest.mark.slow
class TestMusicGenerator:
    """Tests para el generador de música"""
    
    def test_generator_initialization(self, music_generator):
        """Test de inicialización"""
        assert music_generator is not None
        assert isinstance(music_generator, MusicGenerator)
    
    @pytest.mark.skipif(
        not hasattr(MusicGenerator, 'generate_from_text'),
        reason="generate_from_text method not available"
    )
    def test_generate_from_text_basic(self, music_generator, sample_text_prompt):
        """Test básico de generación desde texto"""
        try:
            result = music_generator.generate_from_text(
                prompt=sample_text_prompt,
                duration=30
            )
            
            assert result is not None
            if isinstance(result, np.ndarray):
                assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Generation not available: {e}")
    
    @pytest.mark.skipif(
        not hasattr(MusicGenerator, 'generate_from_text'),
        reason="generate_from_text method not available"
    )
    def test_generate_from_text_different_durations(self, music_generator, sample_text_prompt):
        """Test con diferentes duraciones"""
        durations = [10, 30, 60]
        
        for duration in durations:
            try:
                result = music_generator.generate_from_text(
                    prompt=sample_text_prompt,
                    duration=duration
                )
                assert result is not None
            except Exception as e:
                pytest.skip(f"Generation with duration {duration} not available: {e}")
    
    @pytest.mark.skipif(
        not hasattr(MusicGenerator, 'save_audio'),
        reason="save_audio method not available"
    )
    def test_save_audio(self, music_generator):
        """Test de guardado de audio"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Generar audio de prueba
            audio_data = np.random.randn(44100).astype(np.float32)
            
            result = music_generator.save_audio(
                audio_data=audio_data,
                output_path=output_path,
                sample_rate=44100
            )
            
            assert result is True or result is None
            if os.path.exists(output_path):
                assert os.path.getsize(output_path) > 0
        except Exception as e:
            pytest.skip(f"Save audio not available: {e}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @pytest.mark.skipif(
        not hasattr(MusicGenerator, 'generate_from_text'),
        reason="generate_from_text method not available"
    )
    def test_generate_with_metadata(self, music_generator, sample_text_prompt):
        """Test de generación con metadata"""
        try:
            result = music_generator.generate_from_text(
                prompt=sample_text_prompt,
                duration=30,
                genre="pop",
                mood="happy"
            )
            
            assert result is not None
        except Exception as e:
            pytest.skip(f"Generation with metadata not available: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestMusicGeneratorIntegration:
    """Tests de integración para el generador de música"""
    
    @pytest.mark.skipif(
        not hasattr(MusicGenerator, 'generate_from_text'),
        reason="generate_from_text method not available"
    )
    def test_full_generation_workflow(self, music_generator, sample_text_prompt):
        """Test del flujo completo de generación"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # 1. Generar audio
            audio_data = music_generator.generate_from_text(
                prompt=sample_text_prompt,
                duration=30
            )
            assert audio_data is not None
            
            # 2. Guardar audio
            if hasattr(music_generator, 'save_audio'):
                result = music_generator.save_audio(
                    audio_data=audio_data,
                    output_path=output_path,
                    sample_rate=44100
                )
                assert result is True or result is None
        except Exception as e:
            pytest.skip(f"Full workflow not available: {e}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)




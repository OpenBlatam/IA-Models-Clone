"""
Tests refactorizados para audio processors
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import numpy as np
from services.audio_processors.normalizer import Normalizer
from services.audio_processors.effects_processor import EffectsProcessor
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestNormalizerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para Normalizer"""
    
    @pytest.fixture
    def normalizer(self):
        """Fixture para Normalizer"""
        return Normalizer()
    
    @pytest.mark.parametrize("shape,expected_shape", [
        ((1000,), (1000,)),
        ((2, 1000), (2, 1000)),
        ((44100,), (44100,))
    ])
    def test_normalize_shape(self, normalizer, shape, expected_shape):
        """Test de normalización con diferentes formas"""
        audio = np.random.randn(*shape).astype(np.float32) * 0.5
        normalized = normalizer.normalize(audio)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        assert normalized.shape == expected_shape
    
    @pytest.mark.parametrize("audio_type", [
        "normal",
        "silent",
        "loud"
    ])
    def test_normalize_different_audio(self, normalizer, audio_type):
        """Test de normalización de diferentes tipos de audio"""
        if audio_type == "normal":
            audio = np.random.randn(1000).astype(np.float32) * 0.5
        elif audio_type == "silent":
            audio = np.zeros(1000, dtype=np.float32)
        else:  # loud
            audio = np.ones(1000, dtype=np.float32) * 2.0
        
        normalized = normalizer.normalize(audio)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        if audio_type == "loud":
            # Audio fuerte debería estar normalizado
            assert np.abs(normalized).max() <= 1.0


class TestEffectsProcessorRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para EffectsProcessor"""
    
    @pytest.fixture
    def effects_processor(self):
        """Fixture para EffectsProcessor"""
        return EffectsProcessor()
    
    def test_effects_processor_init(self, effects_processor):
        """Test de inicialización"""
        assert effects_processor is not None
    
    @pytest.mark.parametrize("shape,sample_rate", [
        ((1000,), 44100),
        ((2, 1000), 44100),
        ((44100,), 48000)
    ])
    def test_apply_effects(self, effects_processor, shape, sample_rate):
        """Test de aplicar efectos con diferentes configuraciones"""
        audio = np.random.randn(*shape).astype(np.float32)
        
        processed = effects_processor.apply(audio, sample_rate)
        
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
    
    def test_apply_effects_no_pedalboard(self):
        """Test cuando pedalboard no está disponible"""
        with patch('services.audio_processors.effects_processor.pedalboard', None):
            processor = EffectsProcessor()
            audio = np.random.randn(1000).astype(np.float32)
            
            processed = processor.apply(audio, 44100)
            
            # Debería retornar audio sin procesar
            assert np.array_equal(processed, audio)
    
    def test_is_available(self, effects_processor):
        """Test de disponibilidad"""
        available = effects_processor.is_available()
        
        # Puede ser True o False dependiendo de si pedalboard está instalado
        assert isinstance(available, bool)
    
    def test_apply_effects_error_handling(self, effects_processor):
        """Test de manejo de errores"""
        # Audio inválido
        audio = np.array([1, 2, 3], dtype=np.int32)
        sample_rate = 44100
        
        # No debería lanzar error
        processed = effects_processor.apply(audio, sample_rate)
        
        assert isinstance(processed, np.ndarray)




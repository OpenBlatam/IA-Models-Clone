"""
Tests para audio processors
"""

import pytest
import numpy as np
from services.audio_processors.normalizer import Normalizer
from services.audio_processors.effects_processor import EffectsProcessor


@pytest.mark.unit
@pytest.mark.services
class TestNormalizer:
    """Tests para Normalizer"""
    
    @pytest.fixture
    def normalizer(self):
        """Fixture para Normalizer"""
        return Normalizer()
    
    def test_normalize_mono(self, normalizer):
        """Test de normalización mono"""
        audio = np.random.randn(1000).astype(np.float32) * 0.5
        normalized = normalizer.normalize(audio)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        assert len(normalized) == len(audio)
    
    def test_normalize_stereo(self, normalizer):
        """Test de normalización estéreo"""
        audio = np.random.randn(2, 1000).astype(np.float32) * 0.5
        normalized = normalizer.normalize(audio)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        assert normalized.shape == audio.shape
    
    def test_normalize_silent_audio(self, normalizer):
        """Test de normalización de audio silencioso"""
        audio = np.zeros(1000, dtype=np.float32)
        normalized = normalizer.normalize(audio)
        
        assert isinstance(normalized, np.ndarray)
        assert np.allclose(normalized, audio)
    
    def test_normalize_loud_audio(self, normalizer):
        """Test de normalización de audio fuerte"""
        audio = np.ones(1000, dtype=np.float32) * 2.0
        normalized = normalizer.normalize(audio)
        
        assert isinstance(normalized, np.ndarray)
        # Debería estar normalizado
        assert np.abs(normalized).max() <= 1.0


@pytest.mark.unit
@pytest.mark.services
class TestEffectsProcessor:
    """Tests para EffectsProcessor"""
    
    @pytest.fixture
    def effects_processor(self):
        """Fixture para EffectsProcessor"""
        return EffectsProcessor()
    
    def test_effects_processor_init(self, effects_processor):
        """Test de inicialización"""
        assert effects_processor is not None
    
    def test_apply_effects_mono(self, effects_processor):
        """Test de aplicar efectos a audio mono"""
        audio = np.random.randn(1000).astype(np.float32)
        sample_rate = 44100
        
        processed = effects_processor.apply(audio, sample_rate)
        
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
    
    def test_apply_effects_stereo(self, effects_processor):
        """Test de aplicar efectos a audio estéreo"""
        audio = np.random.randn(2, 1000).astype(np.float32)
        sample_rate = 44100
        
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




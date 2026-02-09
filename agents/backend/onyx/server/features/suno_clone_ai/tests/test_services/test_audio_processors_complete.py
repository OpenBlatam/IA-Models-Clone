"""
Tests completos para todos los audio processors
"""

import pytest
import numpy as np
from unittest.mock import patch

from services.audio_processors.mastering_processor import MasteringProcessor
from services.audio_processors.noise_reducer import NoiseReducer
from services.audio_processors.stereo_converter import StereoConverter
from services.audio_processors.normalizer import Normalizer
from services.audio_processors.effects_processor import EffectsProcessor
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestMasteringProcessor(BaseServiceTestCase, StandardTestMixin):
    """Tests para MasteringProcessor"""
    
    @pytest.fixture
    def mastering_processor(self):
        """Fixture para MasteringProcessor"""
        return MasteringProcessor()
    
    @pytest.fixture
    def mastering_processor_fast(self):
        """Fixture para MasteringProcessor en modo rápido"""
        return MasteringProcessor(fast_mode=True)
    
    @pytest.mark.parametrize("shape,sample_rate", [
        ((1000,), 44100),
        ((2, 1000), 44100),
        ((44100,), 48000)
    ])
    def test_master_mono(self, mastering_processor, shape, sample_rate):
        """Test de mastering con audio mono"""
        audio = np.random.randn(*shape).astype(np.float32) * 0.5
        
        with patch('services.audio_processors.mastering_processor.scipy'):
            result = mastering_processor.master(audio, sample_rate)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
    
    def test_master_fast_mode(self, mastering_processor_fast):
        """Test de mastering en modo rápido"""
        audio = np.random.randn(1000).astype(np.float32)
        
        result = mastering_processor_fast.master(audio, 44100)
        
        # En modo rápido debería retornar el audio sin procesar
        assert np.array_equal(result, audio)
    
    def test_master_stereo(self, mastering_processor):
        """Test de mastering con audio estéreo"""
        audio = np.random.randn(2, 1000).astype(np.float32) * 0.5
        
        with patch('services.audio_processors.mastering_processor.scipy'):
            result = mastering_processor.master(audio, 44100)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.shape == audio.shape
    
    def test_master_error_handling(self, mastering_processor):
        """Test de manejo de errores"""
        audio = np.random.randn(1000).astype(np.float32)
        
        # Simular error en scipy
        with patch('services.audio_processors.mastering_processor.scipy', side_effect=ImportError()):
            result = mastering_processor.master(audio, 44100)
            
            # Debería retornar audio sin procesar en caso de error
            assert isinstance(result, np.ndarray)


class TestNoiseReducer(BaseServiceTestCase, StandardTestMixin):
    """Tests para NoiseReducer"""
    
    @pytest.fixture
    def noise_reducer(self):
        """Fixture para NoiseReducer"""
        return NoiseReducer()
    
    @pytest.mark.parametrize("shape,sample_rate", [
        ((1000,), 44100),
        ((44100,), 48000),
        ((88200,), 44100)
    ])
    def test_reduce_noise(self, noise_reducer, shape, sample_rate):
        """Test de reducción de ruido"""
        audio = np.random.randn(*shape).astype(np.float32)
        
        with patch('services.audio_processors.noise_reducer.nr') as mock_nr:
            mock_nr.reduce_noise.return_value = audio
            result = noise_reducer.reduce(audio, sample_rate)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
    
    def test_reduce_noise_no_library(self, noise_reducer):
        """Test cuando noisereduce no está disponible"""
        audio = np.random.randn(1000).astype(np.float32)
        
        with patch('services.audio_processors.noise_reducer.nr', side_effect=ImportError()):
            result = noise_reducer.reduce(audio, 44100)
            
            # Debería retornar audio sin procesar
            assert np.array_equal(result, audio)
    
    def test_reduce_noise_error_handling(self, noise_reducer):
        """Test de manejo de errores"""
        audio = np.random.randn(1000).astype(np.float32)
        
        with patch('services.audio_processors.noise_reducer.nr') as mock_nr:
            mock_nr.reduce_noise.side_effect = Exception("Error")
            result = noise_reducer.reduce(audio, 44100)
            
            # Debería retornar audio sin procesar en caso de error
            assert np.array_equal(result, audio)


class TestStereoConverter(BaseServiceTestCase, StandardTestMixin):
    """Tests para StereoConverter"""
    
    @pytest.fixture
    def stereo_converter(self):
        """Fixture para StereoConverter"""
        return StereoConverter()
    
    @pytest.fixture
    def stereo_converter_fast(self):
        """Fixture para StereoConverter en modo rápido"""
        return StereoConverter(fast_mode=True)
    
    def test_convert_mono_to_stereo(self, stereo_converter):
        """Test de conversión mono a estéreo"""
        audio = np.random.randn(1000).astype(np.float32)
        
        with patch('services.audio_processors.stereo_converter.scipy'):
            result = stereo_converter.convert(audio)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.ndim == 2
            assert result.shape[0] == 2  # Estéreo
    
    def test_convert_fast_mode(self, stereo_converter_fast):
        """Test de conversión en modo rápido"""
        audio = np.random.randn(1000).astype(np.float32)
        
        result = stereo_converter_fast.convert(audio)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 2
    
    def test_convert_already_stereo(self, stereo_converter):
        """Test cuando el audio ya es estéreo"""
        audio = np.random.randn(2, 1000).astype(np.float32)
        
        result = stereo_converter.convert(audio)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == audio.shape
    
    def test_convert_error_handling(self, stereo_converter):
        """Test de manejo de errores"""
        audio = np.random.randn(1000).astype(np.float32)
        
        with patch('services.audio_processors.stereo_converter.scipy', side_effect=ImportError()):
            result = stereo_converter.convert(audio)
            
            # Debería retornar audio estéreo básico
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2


class TestAllAudioProcessorsIntegration(BaseServiceTestCase, StandardTestMixin):
    """Tests de integración para todos los processors"""
    
    @pytest.fixture
    def processors(self):
        """Fixture con todos los processors"""
        return {
            'normalizer': Normalizer(),
            'effects': EffectsProcessor(),
            'mastering': MasteringProcessor(),
            'noise_reducer': NoiseReducer(),
            'stereo_converter': StereoConverter()
        }
    
    def test_processor_chain(self, processors):
        """Test de cadena de procesamiento"""
        audio = np.random.randn(1000).astype(np.float32) * 0.5
        
        # Normalizar
        with patch('services.audio_processors.mastering_processor.scipy'):
            with patch('services.audio_processors.stereo_converter.scipy'):
                normalized = processors['normalizer'].normalize(audio)
                
                # Convertir a estéreo
                stereo = processors['stereo_converter'].convert(normalized)
                
                # Aplicar efectos
                with patch('services.audio_processors.effects_processor.pedalboard'):
                    processed = processors['effects'].apply(stereo[0], 44100)
                    
                    assert isinstance(processed, np.ndarray)
                    assert processed.dtype == np.float32
    
    @pytest.mark.parametrize("processor_name", [
        'normalizer',
        'effects',
        'mastering',
        'noise_reducer',
        'stereo_converter'
    ])
    def test_processor_handles_empty_audio(self, processors, processor_name):
        """Test de que cada processor maneja audio vacío"""
        audio = np.array([], dtype=np.float32)
        processor = processors[processor_name]
        
        if processor_name == 'normalizer':
            result = processor.normalize(audio)
        elif processor_name == 'effects':
            result = processor.apply(audio, 44100)
        elif processor_name == 'mastering':
            with patch('services.audio_processors.mastering_processor.scipy'):
                result = processor.master(audio, 44100)
        elif processor_name == 'noise_reducer':
            with patch('services.audio_processors.noise_reducer.nr') as mock_nr:
                mock_nr.reduce_noise.return_value = audio
                result = processor.reduce(audio, 44100)
        else:  # stereo_converter
            with patch('services.audio_processors.stereo_converter.scipy'):
                result = processor.convert(audio)
        
        assert isinstance(result, np.ndarray)




"""
Tests para el módulo de procesamiento de audio
"""

import pytest
import numpy as np
import torch
import torchaudio
from pathlib import Path
import tempfile
import os

from core.audio.processing import AudioProcessor, AudioEnhancer


@pytest.fixture
def audio_processor():
    """Fixture para AudioProcessor"""
    return AudioProcessor(sample_rate=44100)


@pytest.fixture
def sample_audio_tensor():
    """Audio tensor de prueba"""
    return torch.randn(1, 44100)  # 1 segundo a 44.1kHz


@pytest.fixture
def sample_audio_array():
    """Audio array de prueba"""
    return np.random.randn(44100).astype(np.float32)


@pytest.fixture
def temp_audio_file(sample_audio_tensor):
    """Archivo de audio temporal"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        torchaudio.save(temp_path, sample_audio_tensor, 44100)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
@pytest.mark.core
class TestAudioProcessor:
    """Tests para AudioProcessor"""
    
    def test_init(self):
        """Test de inicialización"""
        processor = AudioProcessor(sample_rate=48000)
        assert processor.sample_rate == 48000
    
    def test_normalize_tensor(self, audio_processor, sample_audio_tensor):
        """Test de normalización con tensor"""
        normalized = audio_processor.normalize(sample_audio_tensor)
        
        assert isinstance(normalized, torch.Tensor)
        assert torch.abs(normalized).max() <= 1.0
    
    def test_normalize_array(self, audio_processor, sample_audio_array):
        """Test de normalización con array"""
        normalized = audio_processor.normalize(sample_audio_array)
        
        assert isinstance(normalized, np.ndarray)
        assert np.abs(normalized).max() <= 1.0
    
    def test_normalize_with_target_max(self, audio_processor, sample_audio_tensor):
        """Test de normalización con target_max personalizado"""
        normalized = audio_processor.normalize(sample_audio_tensor, target_max=0.5)
        
        assert torch.abs(normalized).max() <= 0.5
    
    def test_normalize_zero_audio(self, audio_processor):
        """Test de normalización con audio en silencio"""
        zero_audio = torch.zeros(1, 1000)
        normalized = audio_processor.normalize(zero_audio)
        
        assert torch.allclose(normalized, zero_audio)
    
    def test_resample_same_rate(self, audio_processor, sample_audio_tensor):
        """Test de resample con misma tasa"""
        resampled = audio_processor.resample(sample_audio_tensor, 44100, 44100)
        
        assert torch.allclose(resampled, sample_audio_tensor)
    
    def test_resample_different_rate(self, audio_processor, sample_audio_tensor):
        """Test de resample con tasa diferente"""
        resampled = audio_processor.resample(sample_audio_tensor, 44100, 22050)
        
        assert resampled.shape[1] < sample_audio_tensor.shape[1]
        assert isinstance(resampled, torch.Tensor)
    
    def test_resample_default_rate(self, audio_processor, sample_audio_tensor):
        """Test de resample usando sample_rate por defecto"""
        resampled = audio_processor.resample(sample_audio_tensor, 22050)
        
        assert isinstance(resampled, torch.Tensor)
    
    def test_trim_silence(self, audio_processor):
        """Test de trim de silencio"""
        # Crear audio con silencio al inicio y final
        audio = np.zeros(1000)
        audio[200:800] = np.random.randn(600) * 0.5
        
        trimmed = audio_processor.trim_silence(audio, threshold=0.01)
        
        assert len(trimmed) <= len(audio)
        assert np.abs(trimmed).max() > 0
    
    def test_trim_silence_all_silent(self, audio_processor):
        """Test de trim con audio completamente silencioso"""
        silent_audio = np.zeros(1000)
        trimmed = audio_processor.trim_silence(silent_audio)
        
        assert len(trimmed) == len(silent_audio)
    
    def test_apply_fade(self, audio_processor):
        """Test de aplicar fade in/out"""
        audio = np.ones(10000)
        
        faded = audio_processor.apply_fade(
            audio,
            fade_in_samples=1000,
            fade_out_samples=1000
        )
        
        assert len(faded) == len(audio)
        # Verificar que el inicio y final tienen valores menores
        assert faded[0] < 1.0
        assert faded[-1] < 1.0
        assert faded[5000] == 1.0  # Medio sin fade
    
    def test_apply_fade_no_fade(self, audio_processor):
        """Test de fade con fade_in/fade_out en 0"""
        audio = np.ones(1000)
        faded = audio_processor.apply_fade(audio, fade_in_samples=0, fade_out_samples=0)
        
        assert np.allclose(faded, audio)
    
    def test_save_audio_tensor(self, audio_processor, sample_audio_tensor):
        """Test de guardar audio desde tensor"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            saved_path = audio_processor.save_audio(sample_audio_tensor, temp_path)
            
            assert os.path.exists(saved_path)
            assert saved_path == temp_path
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_audio_array(self, audio_processor, sample_audio_array):
        """Test de guardar audio desde array"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            saved_path = audio_processor.save_audio(sample_audio_array, temp_path)
            
            assert os.path.exists(saved_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_audio_custom_sample_rate(self, audio_processor, sample_audio_tensor):
        """Test de guardar con sample_rate personalizado"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            saved_path = audio_processor.save_audio(
                sample_audio_tensor,
                temp_path,
                sample_rate=48000
            )
            
            assert os.path.exists(saved_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_audio(self, audio_processor, temp_audio_file):
        """Test de cargar audio"""
        audio, sample_rate = audio_processor.load_audio(temp_audio_file)
        
        assert isinstance(audio, torch.Tensor)
        assert sample_rate == 44100
        assert audio.shape[0] > 0
    
    def test_load_audio_with_resample(self, audio_processor, temp_audio_file):
        """Test de cargar audio con resample"""
        audio, sample_rate = audio_processor.load_audio(
            temp_audio_file,
            target_sample_rate=22050
        )
        
        assert sample_rate == 22050
        assert isinstance(audio, torch.Tensor)


@pytest.mark.unit
@pytest.mark.core
class TestAudioEnhancer:
    """Tests para AudioEnhancer"""
    
    def test_reduce_noise(self):
        """Test de reducción de ruido"""
        audio = np.random.randn(1000).astype(np.float32)
        
        denoised = AudioEnhancer.reduce_noise(audio, noise_reduction_factor=0.5)
        
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == audio.shape
    
    def test_reduce_noise_different_factor(self):
        """Test de reducción con factor diferente"""
        audio = np.random.randn(1000).astype(np.float32)
        
        denoised = AudioEnhancer.reduce_noise(audio, noise_reduction_factor=0.8)
        
        assert isinstance(denoised, np.ndarray)
    
    def test_enhance_quality_all_options(self):
        """Test de mejora de calidad con todas las opciones"""
        audio = np.random.randn(10000).astype(np.float32)
        
        enhanced = AudioEnhancer.enhance_quality(
            audio,
            normalize=True,
            trim_silence=True,
            apply_fade=True
        )
        
        assert isinstance(enhanced, np.ndarray)
        assert len(enhanced) <= len(audio)
    
    def test_enhance_quality_no_options(self):
        """Test de mejora sin opciones"""
        audio = np.random.randn(1000).astype(np.float32)
        
        enhanced = AudioEnhancer.enhance_quality(
            audio,
            normalize=False,
            trim_silence=False,
            apply_fade=False
        )
        
        assert isinstance(enhanced, np.ndarray)
        assert len(enhanced) == len(audio)
    
    def test_enhance_quality_partial_options(self):
        """Test de mejora con opciones parciales"""
        audio = np.random.randn(10000).astype(np.float32)
        
        enhanced = AudioEnhancer.enhance_quality(
            audio,
            normalize=True,
            trim_silence=False,
            apply_fade=True
        )
        
        assert isinstance(enhanced, np.ndarray)




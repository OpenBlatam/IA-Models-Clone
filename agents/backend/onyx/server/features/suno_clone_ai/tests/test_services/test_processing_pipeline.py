"""
Tests para el pipeline de procesamiento
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from services.processing_pipeline import ProcessingPipeline


@pytest.fixture
def mock_audio_processor():
    """Mock del procesador de audio"""
    processor = Mock()
    processor.upsample = Mock(return_value=(np.array([0.1, 0.2, 0.3]), 44100))
    processor.reduce_noise = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.remove_artifacts = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.apply_eq = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.enhance_dynamics = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.apply_saturation = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.convert_to_stereo = Mock(return_value=np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]))
    processor.normalize = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.apply_dithering = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    processor.trim_to_duration = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    return processor


@pytest.fixture
def mock_mastering_processor():
    """Mock del procesador de mastering"""
    processor = Mock()
    processor.master = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    return processor


@pytest.fixture
def mock_effects_processor():
    """Mock del procesador de efectos"""
    processor = Mock()
    processor.is_available = Mock(return_value=True)
    processor.apply = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    return processor


@pytest.fixture
def processing_pipeline(mock_audio_processor, mock_mastering_processor, mock_effects_processor):
    """Instancia del pipeline con mocks"""
    return ProcessingPipeline(
        mock_audio_processor,
        mock_mastering_processor,
        mock_effects_processor,
        fast_mode=False
    )


@pytest.fixture
def processing_pipeline_fast(mock_audio_processor, mock_mastering_processor, mock_effects_processor):
    """Instancia del pipeline en modo rápido"""
    return ProcessingPipeline(
        mock_audio_processor,
        mock_mastering_processor,
        mock_effects_processor,
        fast_mode=True
    )


@pytest.mark.unit
class TestProcessingPipeline:
    """Tests para el pipeline de procesamiento"""
    
    def test_pipeline_initialization(self, processing_pipeline):
        """Test de inicialización"""
        assert processing_pipeline is not None
        assert isinstance(processing_pipeline, ProcessingPipeline)
    
    def test_process_basic(self, processing_pipeline):
        """Test básico de procesamiento"""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        
        result = processing_pipeline.process(
            audio=audio,
            sample_rate=32000,
            duration=30,
            use_noise_reduction=True
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
    
    def test_process_empty_audio(self, processing_pipeline):
        """Test con audio vacío"""
        audio = np.array([], dtype=np.float32)
        
        with pytest.raises(ValueError, match="empty"):
            processing_pipeline.process(
                audio=audio,
                sample_rate=32000,
                duration=30,
                use_noise_reduction=False
            )
    
    def test_process_with_nan_values(self, processing_pipeline):
        """Test con valores NaN"""
        audio = np.array([0.1, np.nan, 0.3, np.inf], dtype=np.float32)
        
        # No debería lanzar excepción, debería limpiar los valores
        result = processing_pipeline.process(
            audio=audio,
            sample_rate=32000,
            duration=30,
            use_noise_reduction=False
        )
        
        assert result is not None
        assert np.all(np.isfinite(result))
    
    def test_process_fast_mode(self, processing_pipeline_fast):
        """Test en modo rápido"""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        result = processing_pipeline_fast.process(
            audio=audio,
            sample_rate=32000,
            duration=30,
            use_noise_reduction=False
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_process_without_effects_processor(self):
        """Test sin procesador de efectos"""
        mock_audio_processor = Mock()
        mock_audio_processor.upsample = Mock(return_value=(np.array([0.1, 0.2]), 44100))
        mock_audio_processor.convert_to_stereo = Mock(return_value=np.array([[0.1, 0.2], [0.1, 0.2]]))
        mock_audio_processor.normalize = Mock(return_value=np.array([0.1, 0.2]))
        mock_audio_processor.trim_to_duration = Mock(return_value=np.array([0.1, 0.2]))
        
        mock_mastering_processor = Mock()
        mock_mastering_processor.master = Mock(return_value=np.array([0.1, 0.2]))
        
        pipeline = ProcessingPipeline(
            mock_audio_processor,
            mock_mastering_processor,
            None,  # Sin effects processor
            fast_mode=True
        )
        
        audio = np.array([0.1, 0.2], dtype=np.float32)
        result = pipeline.process(
            audio=audio,
            sample_rate=32000,
            duration=30,
            use_noise_reduction=False
        )
        
        assert result is not None


@pytest.mark.integration
class TestProcessingPipelineIntegration:
    """Tests de integración para el pipeline"""
    
    def test_full_processing_workflow(self, processing_pipeline):
        """Test del flujo completo de procesamiento"""
        # Audio de entrada
        audio = np.random.randn(44100 * 30).astype(np.float32)
        
        # Procesar
        result = processing_pipeline.process(
            audio=audio,
            sample_rate=32000,
            duration=30,
            use_noise_reduction=True
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))




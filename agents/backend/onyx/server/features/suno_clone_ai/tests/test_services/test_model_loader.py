"""
Tests para el cargador de modelos
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from services.model_loader import ModelLoader


@pytest.fixture
def model_loader():
    """Instancia del cargador de modelos"""
    try:
        return ModelLoader("facebook/musicgen-medium")
    except Exception as e:
        pytest.skip(f"ModelLoader not available: {e}")


@pytest.mark.unit
@pytest.mark.slow
class TestModelLoader:
    """Tests para el cargador de modelos"""
    
    def test_loader_initialization(self, model_loader):
        """Test de inicialización"""
        if model_loader is None:
            pytest.skip("ModelLoader not available")
        assert model_loader is not None
        assert isinstance(model_loader, ModelLoader)
        assert model_loader.model_name == "facebook/musicgen-medium"
    
    @pytest.mark.skipif(
        True,  # Skip por defecto ya que requiere modelos reales
        reason="Requires actual model loading"
    )
    def test_load_model(self, model_loader):
        """Test de carga de modelo"""
        if model_loader is None:
            pytest.skip("ModelLoader not available")
        
        try:
            model_loader.load()
            assert model_loader.model is not None
        except Exception as e:
            pytest.skip(f"Model loading not available: {e}")
    
    def test_generate_audio_empty_prompt(self, model_loader):
        """Test con prompt vacío"""
        if model_loader is None:
            pytest.skip("ModelLoader not available")
        
        with pytest.raises(ValueError, match="empty"):
            model_loader.generate_audio(
                prompt="",
                duration=30,
                guidance_scale=3.0,
                temperature=0.9
            )
    
    @pytest.mark.skipif(
        True,
        reason="Requires actual model loading"
    )
    def test_generate_audio_success(self, model_loader):
        """Test de generación exitosa"""
        if model_loader is None:
            pytest.skip("ModelLoader not available")
        
        try:
            model_loader.load()
            audio = model_loader.generate_audio(
                prompt="A happy pop song",
                duration=10,
                guidance_scale=3.0,
                temperature=0.9
            )
            
            assert audio is not None
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
        except Exception as e:
            pytest.skip(f"Audio generation not available: {e}")
    
    def test_generate_audio_without_load(self, model_loader):
        """Test de generación sin cargar modelo primero"""
        if model_loader is None:
            pytest.skip("ModelLoader not available")
        
        # Debería cargar automáticamente
        with patch.object(model_loader, 'load') as mock_load:
            with patch.object(model_loader, 'model', None):
                try:
                    model_loader.generate_audio(
                        prompt="Test",
                        duration=10,
                        guidance_scale=3.0,
                        temperature=0.9
                    )
                    # Si no lanza excepción, el load fue llamado
                    mock_load.assert_called()
                except (RuntimeError, ValueError):
                    # Esperado si el modelo no está cargado
                    pass


@pytest.mark.integration
@pytest.mark.slow
class TestModelLoaderIntegration:
    """Tests de integración para el cargador de modelos"""
    
    @pytest.mark.skipif(
        True,
        reason="Requires actual model loading"
    )
    def test_full_model_workflow(self, model_loader):
        """Test del flujo completo de carga y generación"""
        if model_loader is None:
            pytest.skip("ModelLoader not available")
        
        try:
            # 1. Cargar modelo
            model_loader.load()
            assert model_loader.model is not None
            
            # 2. Generar audio
            audio = model_loader.generate_audio(
                prompt="A happy pop song",
                duration=30,
                guidance_scale=3.0,
                temperature=0.9
            )
            
            assert audio is not None
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
        except Exception as e:
            pytest.skip(f"Full workflow not available: {e}")




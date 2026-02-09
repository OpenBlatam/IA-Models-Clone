"""
Tests para utilidades de modelos
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from core.utils.model_utils import (
    compile_model,
    enable_gradient_checkpointing,
    count_parameters,
    get_model_size_mb
)


@pytest.fixture
def sample_model():
    """Modelo simple para tests"""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )


@pytest.mark.unit
@pytest.mark.core
class TestCompileModel:
    """Tests para compile_model"""
    
    def test_compile_model_with_torch_compile(self, sample_model):
        """Test de compilación con torch.compile disponible"""
        with patch('torch.compile', return_value=sample_model) as mock_compile:
            result = compile_model(sample_model)
            
            mock_compile.assert_called_once()
            assert result is not None
    
    def test_compile_model_default_mode(self, sample_model):
        """Test de compilación con modo por defecto"""
        with patch('torch.compile', return_value=sample_model) as mock_compile:
            result = compile_model(sample_model, mode="default")
            
            mock_compile.assert_called_once_with(
                sample_model,
                mode="default",
                fullgraph=False
            )
    
    def test_compile_model_custom_mode(self, sample_model):
        """Test de compilación con modo personalizado"""
        with patch('torch.compile', return_value=sample_model) as mock_compile:
            result = compile_model(sample_model, mode="max-autotune", fullgraph=True)
            
            mock_compile.assert_called_once_with(
                sample_model,
                mode="max-autotune",
                fullgraph=True
            )
    
    def test_compile_model_no_torch_compile(self, sample_model):
        """Test de compilación sin torch.compile"""
        with patch('hasattr', return_value=False):
            result = compile_model(sample_model)
            
            # Debería retornar el modelo original
            assert result == sample_model
    
    def test_compile_model_compilation_fails(self, sample_model):
        """Test cuando la compilación falla"""
        with patch('torch.compile', side_effect=Exception("Compilation failed")):
            result = compile_model(sample_model)
            
            # Debería retornar el modelo original
            assert result == sample_model


@pytest.mark.unit
@pytest.mark.core
class TestEnableGradientCheckpointing:
    """Tests para enable_gradient_checkpointing"""
    
    def test_enable_gradient_checkpointing_with_method(self, sample_model):
        """Test de habilitación cuando el modelo tiene el método"""
        sample_model.gradient_checkpointing_enable = MagicMock()
        
        enable_gradient_checkpointing(sample_model)
        
        sample_model.gradient_checkpointing_enable.assert_called_once()
    
    def test_enable_gradient_checkpointing_without_method(self, sample_model):
        """Test de habilitación cuando el modelo no tiene el método"""
        # No debería lanzar error
        enable_gradient_checkpointing(sample_model)
    
    def test_enable_gradient_checkpointing_with_submodule(self):
        """Test de habilitación con submódulo que tiene el método"""
        class ModuleWithCheckpointing(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def gradient_checkpointing_enable(self):
                pass
        
        model = nn.Sequential(
            ModuleWithCheckpointing(),
            nn.Linear(5, 1)
        )
        
        # No debería lanzar error
        enable_gradient_checkpointing(model)


@pytest.mark.unit
@pytest.mark.core
class TestCountParameters:
    """Tests para count_parameters"""
    
    def test_count_parameters_trainable_only(self, sample_model):
        """Test de conteo solo de parámetros entrenables"""
        count = count_parameters(sample_model, trainable_only=True)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_parameters_all(self, sample_model):
        """Test de conteo de todos los parámetros"""
        count_all = count_parameters(sample_model, trainable_only=False)
        count_trainable = count_parameters(sample_model, trainable_only=True)
        
        assert count_all >= count_trainable
    
    def test_count_parameters_with_frozen(self):
        """Test de conteo con parámetros congelados"""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 1)
        )
        
        # Congelar la primera capa
        for param in model[0].parameters():
            param.requires_grad = False
        
        count_trainable = count_parameters(model, trainable_only=True)
        count_all = count_parameters(model, trainable_only=False)
        
        assert count_trainable < count_all


@pytest.mark.unit
@pytest.mark.core
class TestGetModelSizeMB:
    """Tests para get_model_size_mb"""
    
    def test_get_model_size_mb(self, sample_model):
        """Test de obtención de tamaño del modelo"""
        size = get_model_size_mb(sample_model)
        
        assert size > 0
        assert isinstance(size, float)
    
    def test_get_model_size_mb_empty_model(self):
        """Test de tamaño con modelo vacío"""
        model = nn.Module()
        size = get_model_size_mb(model)
        
        assert size >= 0




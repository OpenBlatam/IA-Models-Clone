"""
Tests para utilidades de mixed precision
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from core.utils.mixed_precision import MixedPrecisionManager


@pytest.fixture
def sample_model():
    """Modelo simple para tests"""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )


@pytest.fixture
def sample_optimizer(sample_model):
    """Optimizador simple para tests"""
    return torch.optim.SGD(sample_model.parameters(), lr=0.01)


@pytest.mark.unit
@pytest.mark.core
class TestMixedPrecisionManager:
    """Tests para MixedPrecisionManager"""
    
    @patch('torch.cuda.is_available')
    def test_init_enabled_with_gpu(self, mock_cuda_available):
        """Test de inicialización habilitada con GPU"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        
        assert manager.enabled is True
        assert manager.scaler is not None
    
    @patch('torch.cuda.is_available')
    def test_init_enabled_without_gpu(self, mock_cuda_available):
        """Test de inicialización habilitada sin GPU"""
        mock_cuda_available.return_value = False
        
        manager = MixedPrecisionManager(enabled=True)
        
        assert manager.enabled is False
        assert manager.scaler is None
    
    @patch('torch.cuda.is_available')
    def test_init_disabled(self, mock_cuda_available):
        """Test de inicialización deshabilitada"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=False)
        
        assert manager.enabled is False
        assert manager.scaler is None
    
    def test_init_custom_params(self):
        """Test de inicialización con parámetros personalizados"""
        with patch('torch.cuda.is_available', return_value=True):
            manager = MixedPrecisionManager(
                enabled=True,
                init_scale=32768.0,
                growth_factor=1.5,
                backoff_factor=0.25,
                growth_interval=1000
            )
            
            assert manager.enabled is True
            assert manager.scaler is not None
    
    @patch('torch.cuda.is_available')
    def test_autocast_enabled(self, mock_cuda_available):
        """Test de autocast cuando está habilitado"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        context = manager.autocast()
        
        assert context is not None
    
    @patch('torch.cuda.is_available')
    def test_autocast_disabled(self, mock_cuda_available):
        """Test de autocast cuando está deshabilitado"""
        mock_cuda_available.return_value = False
        
        manager = MixedPrecisionManager(enabled=True)
        context = manager.autocast()
        
        # Debería retornar un nullcontext
        assert context is not None
    
    @patch('torch.cuda.is_available')
    def test_scale_loss_enabled(self, mock_cuda_available):
        """Test de scale_loss cuando está habilitado"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        loss = torch.tensor(1.0)
        
        scaled_loss = manager.scale_loss(loss)
        
        assert scaled_loss is not None
        assert scaled_loss != loss  # Debería estar escalado
    
    @patch('torch.cuda.is_available')
    def test_scale_loss_disabled(self, mock_cuda_available):
        """Test de scale_loss cuando está deshabilitado"""
        mock_cuda_available.return_value = False
        
        manager = MixedPrecisionManager(enabled=True)
        loss = torch.tensor(1.0)
        
        scaled_loss = manager.scale_loss(loss)
        
        assert scaled_loss == loss  # No debería estar escalado
    
    @patch('torch.cuda.is_available')
    def test_step_with_loss(self, mock_cuda_available, sample_model, sample_optimizer):
        """Test de step con loss"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        loss = torch.tensor(1.0, requires_grad=True)
        
        # No debería lanzar error
        manager.step(sample_optimizer, loss)
    
    @patch('torch.cuda.is_available')
    def test_step_without_loss(self, mock_cuda_available, sample_optimizer):
        """Test de step sin loss"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        
        # No debería lanzar error
        manager.step(sample_optimizer)
    
    @patch('torch.cuda.is_available')
    def test_step_disabled(self, mock_cuda_available, sample_model, sample_optimizer):
        """Test de step cuando está deshabilitado"""
        mock_cuda_available.return_value = False
        
        manager = MixedPrecisionManager(enabled=True)
        loss = torch.tensor(1.0, requires_grad=True)
        
        # No debería lanzar error
        manager.step(sample_optimizer, loss)
    
    @patch('torch.cuda.is_available')
    def test_update_scale(self, mock_cuda_available):
        """Test de update_scale"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        
        # No debería lanzar error
        manager.update_scale()
    
    @patch('torch.cuda.is_available')
    def test_get_scale(self, mock_cuda_available):
        """Test de get_scale"""
        mock_cuda_available.return_value = True
        
        manager = MixedPrecisionManager(enabled=True)
        scale = manager.get_scale()
        
        assert scale is not None
        assert isinstance(scale, float)
    
    @patch('torch.cuda.is_available')
    def test_get_scale_disabled(self, mock_cuda_available):
        """Test de get_scale cuando está deshabilitado"""
        mock_cuda_available.return_value = False
        
        manager = MixedPrecisionManager(enabled=True)
        scale = manager.get_scale()
        
        assert scale == 1.0  # Escala por defecto cuando está deshabilitado




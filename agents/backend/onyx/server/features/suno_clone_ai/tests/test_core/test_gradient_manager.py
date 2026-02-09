"""
Tests para gradient manager
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from core.utils.gradients.gradient_manager import (
    clip_gradients,
    get_gradient_norm,
    check_gradients
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class SimpleModel(nn.Module):
    """Modelo simple para tests"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestClipGradients(BaseServiceTestCase, StandardTestMixin):
    """Tests para clip_gradients"""
    
    @pytest.fixture
    def model(self):
        """Fixture para modelo"""
        return SimpleModel()
    
    @pytest.fixture
    def model_with_gradients(self, model):
        """Fixture para modelo con gradientes"""
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        return model
    
    def test_clip_gradients_success(self, model_with_gradients):
        """Test de clipping exitoso de gradientes"""
        max_norm = 1.0
        
        total_norm = clip_gradients(model_with_gradients, max_norm=max_norm)
        
        assert isinstance(total_norm, float)
        assert total_norm >= 0.0
    
    @pytest.mark.parametrize("max_norm", [0.5, 1.0, 2.0, 5.0])
    def test_clip_gradients_different_norms(self, model_with_gradients, max_norm):
        """Test de clipping con diferentes normas máximas"""
        total_norm = clip_gradients(model_with_gradients, max_norm=max_norm)
        
        assert isinstance(total_norm, float)
        assert total_norm >= 0.0
    
    def test_clip_gradients_no_parameters(self):
        """Test cuando el modelo no tiene parámetros"""
        model = nn.Module()
        
        total_norm = clip_gradients(model, max_norm=1.0)
        
        assert total_norm == 0.0


class TestGetGradientNorm(BaseServiceTestCase, StandardTestMixin):
    """Tests para get_gradient_norm"""
    
    @pytest.fixture
    def model(self):
        """Fixture para modelo"""
        return SimpleModel()
    
    @pytest.fixture
    def model_with_gradients(self, model):
        """Fixture para modelo con gradientes"""
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        return model
    
    def test_get_gradient_norm_success(self, model_with_gradients):
        """Test de obtención exitosa de norma de gradientes"""
        norm = get_gradient_norm(model_with_gradients)
        
        assert isinstance(norm, float)
        assert norm >= 0.0
    
    @pytest.mark.parametrize("norm_type", [1.0, 2.0, float('inf')])
    def test_get_gradient_norm_different_types(self, model_with_gradients, norm_type):
        """Test de obtención con diferentes tipos de norma"""
        norm = get_gradient_norm(model_with_gradients, norm_type=norm_type)
        
        assert isinstance(norm, float)
        assert norm >= 0.0
    
    def test_get_gradient_norm_no_parameters(self):
        """Test cuando el modelo no tiene parámetros"""
        model = nn.Module()
        
        norm = get_gradient_norm(model)
        
        assert norm == 0.0
    
    def test_get_gradient_norm_no_gradients(self, model):
        """Test cuando no hay gradientes"""
        norm = get_gradient_norm(model)
        
        assert norm == 0.0


class TestCheckGradients(BaseServiceTestCase, StandardTestMixin):
    """Tests para check_gradients"""
    
    @pytest.fixture
    def model(self):
        """Fixture para modelo"""
        return SimpleModel()
    
    @pytest.fixture
    def model_with_gradients(self, model):
        """Fixture para modelo con gradientes"""
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        return model
    
    def test_check_gradients_success(self, model_with_gradients):
        """Test de verificación exitosa de gradientes"""
        results = check_gradients(model_with_gradients)
        
        assert isinstance(results, dict)
        assert "has_nan" in results
        assert "has_inf" in results
        assert "has_zero" in results
        assert "num_params" in results
        assert "num_params_with_grad" in results
        assert results["num_params"] > 0
    
    @pytest.mark.parametrize("check_nan,check_inf,check_zero", [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True)
    ])
    def test_check_gradients_different_checks(self, model_with_gradients, check_nan, check_inf, check_zero):
        """Test de verificación con diferentes opciones"""
        results = check_gradients(
            model_with_gradients,
            check_nan=check_nan,
            check_inf=check_inf,
            check_zero=check_zero
        )
        
        assert isinstance(results, dict)
        assert isinstance(results["has_nan"], bool)
        assert isinstance(results["has_inf"], bool)
        assert isinstance(results["has_zero"], bool)
    
    def test_check_gradients_no_parameters(self):
        """Test cuando el modelo no tiene parámetros"""
        model = nn.Module()
        
        results = check_gradients(model)
        
        assert results["num_params"] == 0
        assert results["num_params_with_grad"] == 0
    
    def test_check_gradients_no_gradients(self, model):
        """Test cuando no hay gradientes"""
        results = check_gradients(model)
        
        assert results["num_params_with_grad"] == 0




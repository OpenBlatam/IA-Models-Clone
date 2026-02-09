"""
Tests para el validador de tensores
"""

import pytest
import torch
import numpy as np

from core.utils.validation.tensor_validator import (
    check_for_nan_inf,
    validate_tensor
)


@pytest.mark.unit
@pytest.mark.core
class TestCheckForNaNInf:
    """Tests para check_for_nan_inf"""
    
    def test_check_for_nan_inf_no_issues(self):
        """Test de verificación sin problemas"""
        tensor = torch.randn(10, 10)
        
        result = check_for_nan_inf(tensor, "test_tensor")
        
        assert result is False
    
    def test_check_for_nan_inf_with_nan(self):
        """Test de verificación con NaN"""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        result = check_for_nan_inf(tensor, "test_tensor")
        
        assert result is True
    
    def test_check_for_nan_inf_with_inf(self):
        """Test de verificación con Inf"""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        result = check_for_nan_inf(tensor, "test_tensor")
        
        assert result is True
    
    def test_check_for_nan_inf_with_nan_raise(self):
        """Test de verificación con NaN y raise_on_error"""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        with pytest.raises(ValueError, match="NaN detected"):
            check_for_nan_inf(tensor, "test_tensor", raise_on_error=True)
    
    def test_check_for_nan_inf_with_inf_raise(self):
        """Test de verificación con Inf y raise_on_error"""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        with pytest.raises(ValueError, match="Inf detected"):
            check_for_nan_inf(tensor, "test_tensor", raise_on_error=True)
    
    def test_check_for_nan_inf_empty_tensor(self):
        """Test de verificación con tensor vacío"""
        tensor = torch.tensor([])
        
        result = check_for_nan_inf(tensor, "test_tensor")
        
        assert result is False


@pytest.mark.unit
@pytest.mark.core
class TestValidateTensor:
    """Tests para validate_tensor"""
    
    def test_validate_tensor_valid(self):
        """Test de validación de tensor válido"""
        tensor = torch.randn(10, 10)
        
        result = validate_tensor(tensor, "test_tensor")
        
        assert result is True
    
    def test_validate_tensor_with_nan(self):
        """Test de validación con NaN"""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        result = validate_tensor(tensor, "test_tensor", check_nan=True)
        
        assert result is False
    
    def test_validate_tensor_with_inf(self):
        """Test de validación con Inf"""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        result = validate_tensor(tensor, "test_tensor", check_inf=True)
        
        assert result is False
    
    def test_validate_tensor_correct_shape(self):
        """Test de validación con forma correcta"""
        tensor = torch.randn(10, 10)
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_shape=(10, 10)
        )
        
        assert result is True
    
    def test_validate_tensor_incorrect_shape(self):
        """Test de validación con forma incorrecta"""
        tensor = torch.randn(10, 10)
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_shape=(5, 5)
        )
        
        assert result is False
    
    def test_validate_tensor_correct_range(self):
        """Test de validación con rango correcto"""
        tensor = torch.randn(10, 10) * 0.5  # Valores entre -1 y 1 aproximadamente
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_range=(-2.0, 2.0)
        )
        
        assert result is True
    
    def test_validate_tensor_out_of_range(self):
        """Test de validación con valores fuera de rango"""
        tensor = torch.tensor([1.0, 5.0, 3.0])
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_range=(0.0, 4.0)
        )
        
        assert result is False
    
    def test_validate_tensor_all_checks(self):
        """Test de validación con todas las verificaciones"""
        tensor = torch.randn(10, 10)
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_nan=True,
            check_inf=True,
            check_shape=(10, 10),
            check_range=(-5.0, 5.0)
        )
        
        assert result is True
    
    def test_validate_tensor_no_checks(self):
        """Test de validación sin verificaciones"""
        tensor = torch.randn(10, 10)
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_nan=False,
            check_inf=False
        )
        
        assert result is True
    
    def test_validate_tensor_nan_check_disabled(self):
        """Test de validación con check_nan deshabilitado"""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        result = validate_tensor(
            tensor,
            "test_tensor",
            check_nan=False,
            check_inf=True
        )
        
        # Debería pasar porque check_nan está deshabilitado
        assert result is True




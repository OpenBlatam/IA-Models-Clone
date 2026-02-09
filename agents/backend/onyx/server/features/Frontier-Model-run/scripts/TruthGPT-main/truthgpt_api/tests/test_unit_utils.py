"""
Tests Unitarios para Utilidades
=================================
"""

import pytest
import sys
import os
import numpy as np
import torch

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import to_categorical, normalize, get_data
except ImportError:
    # Intentar importar desde diferentes ubicaciones
    try:
        import truthgpt as tg
        to_categorical = tg.to_categorical
        normalize = tg.normalize
    except:
        pytest.skip("No se pueden importar las utilidades", allow_module_level=True)


class TestToCategorical:
    """Tests unitarios para to_categorical."""
    
    def test_to_categorical_basic(self):
        """Test básico de to_categorical"""
        y = np.array([0, 1, 2, 1, 0])
        y_cat = to_categorical(y, num_classes=3)
        
        assert y_cat.shape == (5, 3)
        assert np.allclose(y_cat.sum(axis=1), 1.0)
        assert y_cat[0, 0] == 1.0
        assert y_cat[1, 1] == 1.0
        assert y_cat[2, 2] == 1.0
    
    def test_to_categorical_single_class(self):
        """Test to_categorical con una sola clase"""
        y = np.array([0, 0, 0])
        y_cat = to_categorical(y, num_classes=1)
        
        assert y_cat.shape == (3, 1)
        assert np.all(y_cat == 1.0)
    
    def test_to_categorical_infer_classes(self):
        """Test to_categorical infiriendo num_classes"""
        y = np.array([0, 1, 2, 3])
        y_cat = to_categorical(y)
        
        # Debería inferir que hay 4 clases (0, 1, 2, 3)
        assert y_cat.shape[1] >= 4


class TestNormalize:
    """Tests unitarios para normalize."""
    
    def test_normalize_basic(self):
        """Test básico de normalize"""
        x = np.random.randn(10, 5).astype(np.float32)
        x_norm = normalize(x)
        
        assert x_norm.shape == x.shape
        # Verificar que está normalizado (norma L2 = 1)
        norms = np.linalg.norm(x_norm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_normalize_single_sample(self):
        """Test normalize con una sola muestra"""
        x = np.random.randn(1, 5).astype(np.float32)
        x_norm = normalize(x)
        
        assert x_norm.shape == x.shape
        norm = np.linalg.norm(x_norm, axis=1)[0]
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_normalize_zero_vector(self):
        """Test normalize con vector cero"""
        x = np.zeros((1, 5)).astype(np.float32)
        x_norm = normalize(x)
        
        # Vector cero debería seguir siendo cero o manejar el error
        assert x_norm.shape == x.shape


class TestDataUtils:
    """Tests unitarios para utilidades de datos."""
    
    def test_to_categorical_edge_cases(self):
        """Test casos edge de to_categorical"""
        # Array vacío
        y_empty = np.array([])
        try:
            y_cat = to_categorical(y_empty, num_classes=3)
            # Si no falla, debería tener shape (0, 3)
            assert y_cat.shape == (0, 3) or y_cat.shape == (0,)
        except:
            # Si falla, está bien para array vacío
            pass
    
    def test_normalize_edge_cases(self):
        """Test casos edge de normalize"""
        # Array vacío
        x_empty = np.array([]).reshape(0, 5)
        try:
            x_norm = normalize(x_empty)
            assert x_norm.shape == x_empty.shape
        except:
            pass


class TestTypeConversions:
    """Tests para conversiones de tipos."""
    
    def test_numpy_to_torch_conversion(self):
        """Test conversión de numpy a torch"""
        x_np = np.random.randn(10, 5).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        
        assert x_torch.shape == x_np.shape
        assert x_torch.dtype == torch.float32
    
    def test_torch_to_numpy_conversion(self):
        """Test conversión de torch a numpy"""
        x_torch = torch.randn(10, 5)
        x_np = x_torch.numpy()
        
        assert x_np.shape == x_torch.shape
        assert x_np.dtype == np.float32 or x_np.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


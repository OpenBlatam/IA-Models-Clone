"""
Tests Unitarios para Helpers de la API
=======================================
Tests que prueban funciones helper sin requerir servidor
"""

import pytest
import sys
import os
import json
import numpy as np

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.services import (
        create_layer_from_config,
        create_optimizer_from_config,
        create_loss_from_config
    )
    from api.schemas import LayerConfig
except ImportError:
    # Si no se puede importar, intentar importar desde el módulo directamente
    try:
        from api_server import (
            create_layer_from_config,
            create_optimizer_from_config,
            create_loss_from_config
        )
        from api_server import LayerConfig
    except:
        pytest.skip("No se puede importar api.services", allow_module_level=True)


class TestLayerCreation:
    """Tests para creación de layers desde configuración."""
    
    def test_create_dense_layer(self):
        """Test crear Dense layer desde config"""
        try:
            config = LayerConfig(type='dense', params={'units': 64, 'activation': 'relu'})
        except NameError:
            config = type('LayerConfig', (), {
                'type': 'dense',
                'params': {'units': 64, 'activation': 'relu'}
            })()
        
        layer = create_layer_from_config(config)
        assert layer is not None
        assert hasattr(layer, 'units') or hasattr(layer, 'out_features')
    
    def test_create_conv2d_layer(self):
        """Test crear Conv2D layer desde config"""
        try:
            config = LayerConfig(type='conv2d', params={'filters': 32, 'kernel_size': 3})
        except NameError:
            config = type('LayerConfig', (), {
                'type': 'conv2d',
                'params': {'filters': 32, 'kernel_size': 3}
            })()
        
        layer = create_layer_from_config(config)
        assert layer is not None
    
    def test_create_dropout_layer(self):
        """Test crear Dropout layer desde config"""
        try:
            config = LayerConfig(type='dropout', params={'rate': 0.5})
        except NameError:
            config = type('LayerConfig', (), {
                'type': 'dropout',
                'params': {'rate': 0.5}
            })()
        
        layer = create_layer_from_config(config)
        assert layer is not None
    
    def test_create_invalid_layer(self):
        """Test crear layer inválido"""
        try:
            config = LayerConfig(type='invalid_layer', params={})
        except NameError:
            config = type('LayerConfig', (), {
                'type': 'invalid_layer',
                'params': {}
            })()
        
        from api.exceptions import InvalidLayerTypeError
        with pytest.raises((ValueError, InvalidLayerTypeError)):
            create_layer_from_config(config)


class TestOptimizerCreation:
    """Tests para creación de optimizers desde configuración."""
    
    def test_create_adam_optimizer(self):
        """Test crear Adam optimizer desde config"""
        optimizer = create_optimizer_from_config(
            'adam',
            {'learning_rate': 0.001}
        )
        assert optimizer is not None
    
    def test_create_sgd_optimizer(self):
        """Test crear SGD optimizer desde config"""
        optimizer = create_optimizer_from_config(
            'sgd',
            {'learning_rate': 0.01, 'momentum': 0.9}
        )
        assert optimizer is not None
    
    def test_create_rmsprop_optimizer(self):
        """Test crear RMSprop optimizer desde config"""
        optimizer = create_optimizer_from_config(
            'rmsprop',
            {'learning_rate': 0.001}
        )
        assert optimizer is not None
    
    def test_create_invalid_optimizer(self):
        """Test crear optimizer inválido"""
        from api.exceptions import InvalidOptimizerError
        with pytest.raises((ValueError, InvalidOptimizerError)):
            create_optimizer_from_config('invalid_optimizer', {})


class TestLossCreation:
    """Tests para creación de loss functions desde configuración."""
    
    def test_create_sparse_categorical_crossentropy(self):
        """Test crear SparseCategoricalCrossentropy desde config"""
        loss = create_loss_from_config(
            'sparsecategoricalcrossentropy',
            {}
        )
        assert loss is not None
    
    def test_create_mse_loss(self):
        """Test crear MSE loss desde config"""
        loss = create_loss_from_config(
            'mse',
            {}
        )
        assert loss is not None
    
    def test_create_mae_loss(self):
        """Test crear MAE loss desde config"""
        loss = create_loss_from_config(
            'mae',
            {}
        )
        assert loss is not None
    
    def test_create_invalid_loss(self):
        """Test crear loss inválido"""
        from api.exceptions import InvalidLossError
        with pytest.raises((ValueError, InvalidLossError)):
            create_loss_from_config('invalid_loss', {})


class TestJSONSerialization:
    """Tests para serialización JSON."""
    
    def test_serialize_numpy_array(self):
        """Test serializar numpy array a JSON"""
        arr = np.array([1, 2, 3])
        arr_list = arr.tolist()
        json_str = json.dumps(arr_list)
        assert json_str is not None
    
    def test_deserialize_json_to_array(self):
        """Test deserializar JSON a numpy array"""
        json_str = '[1, 2, 3]'
        arr_list = json.loads(json_str)
        arr = np.array(arr_list)
        assert arr.shape == (3,)
    
    def test_serialize_model_history(self):
        """Test serializar history del modelo"""
        history = {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.7, 0.8, 0.9],
            'val_loss': [0.6, 0.5, 0.4]
        }
        
        # Convertir a formato serializable
        serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                serializable[key] = [float(v) for v in value]
            else:
                serializable[key] = float(value) if isinstance(value, (int, float)) else value
        
        json_str = json.dumps(serializable)
        assert json_str is not None
        
        # Deserializar
        deserialized = json.loads(json_str)
        assert deserialized['loss'] == [0.5, 0.4, 0.3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])












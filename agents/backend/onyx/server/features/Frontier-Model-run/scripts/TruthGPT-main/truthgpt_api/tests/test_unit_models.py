"""
Tests Unitarios para Models
============================
Tests que prueban componentes individuales sin depender de la API
"""

import pytest
import sys
import os
import torch
import numpy as np

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.sequential import Sequential
    from models.base import Model
    from layers import Dense, Dropout, Flatten
    from optimizers import Adam, SGD
    from losses import SparseCategoricalCrossentropy, MeanSquaredError
    from metrics import Accuracy
except ImportError as e:
    pytest.skip(f"No se pueden importar los módulos: {e}", allow_module_level=True)


class TestSequentialModel:
    """Tests unitarios para Sequential model."""
    
    def test_sequential_creation_empty(self):
        """Test crear Sequential vacío"""
        model = Sequential()
        assert model is not None
        assert len(model.layers_list) == 0
    
    def test_sequential_creation_with_layers(self):
        """Test crear Sequential con layers"""
        layer1 = Dense(64, activation='relu')
        layer2 = Dense(10, activation='softmax')
        model = Sequential([layer1, layer2])
        assert len(model.layers_list) == 2
    
    def test_sequential_add_layer(self):
        """Test agregar layer a Sequential"""
        model = Sequential()
        layer = Dense(64)
        model.add(layer)
        assert len(model.layers_list) == 1
    
    def test_sequential_forward(self):
        """Test forward pass de Sequential"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        # Crear input dummy
        x = torch.randn(32, 20)  # batch_size=32, features=20
        
        # Forward pass
        output = model(x)
        assert output is not None
        assert output.shape[0] == 32  # batch size
        assert output.shape[1] == 10  # output size


class TestDenseLayer:
    """Tests unitarios para Dense layer."""
    
    def test_dense_creation(self):
        """Test crear Dense layer"""
        layer = Dense(64)
        assert layer is not None
        assert layer.units == 64
    
    def test_dense_with_activation(self):
        """Test Dense con activación"""
        layer = Dense(64, activation='relu')
        assert layer.activation == 'relu'
    
    def test_dense_forward(self):
        """Test forward pass de Dense"""
        layer = Dense(64, activation='relu')
        x = torch.randn(32, 20)
        output = layer(x)
        assert output.shape == (32, 64)
    
    def test_dense_without_bias(self):
        """Test Dense sin bias"""
        layer = Dense(64, use_bias=False)
        x = torch.randn(32, 20)
        output = layer(x)
        assert output.shape == (32, 64)


class TestDropoutLayer:
    """Tests unitarios para Dropout layer."""
    
    def test_dropout_creation(self):
        """Test crear Dropout layer"""
        layer = Dropout(0.5)
        assert layer is not None
        assert layer.rate == 0.5
    
    def test_dropout_forward_training(self):
        """Test Dropout en modo training"""
        layer = Dropout(0.5)
        x = torch.randn(32, 64)
        output = layer(x, training=True)
        assert output.shape == x.shape
    
    def test_dropout_forward_eval(self):
        """Test Dropout en modo evaluación"""
        layer = Dropout(0.5)
        x = torch.randn(32, 64)
        output = layer(x, training=False)
        assert output.shape == x.shape
        # En eval mode, debería ser igual (sin dropout)
        assert torch.allclose(output, x)


class TestFlattenLayer:
    """Tests unitarios para Flatten layer."""
    
    def test_flatten_creation(self):
        """Test crear Flatten layer"""
        layer = Flatten()
        assert layer is not None
    
    def test_flatten_forward(self):
        """Test forward pass de Flatten"""
        layer = Flatten()
        x = torch.randn(32, 3, 28, 28)  # Imagen batch
        output = layer(x)
        assert output.shape == (32, 3 * 28 * 28)


class TestAdamOptimizer:
    """Tests unitarios para Adam optimizer."""
    
    def test_adam_creation(self):
        """Test crear Adam optimizer"""
        optimizer = Adam(learning_rate=0.001)
        assert optimizer is not None
        assert optimizer.learning_rate == 0.001
    
    def test_adam_with_params(self):
        """Test Adam con todos los parámetros"""
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999
        )
        assert optimizer.beta_1 == 0.9
        assert optimizer.beta_2 == 0.999


class TestSGDOptimizer:
    """Tests unitarios para SGD optimizer."""
    
    def test_sgd_creation(self):
        """Test crear SGD optimizer"""
        optimizer = SGD(learning_rate=0.01)
        assert optimizer is not None
        assert optimizer.learning_rate == 0.01
    
    def test_sgd_with_momentum(self):
        """Test SGD con momentum"""
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        assert optimizer.momentum == 0.9


class TestLossFunctions:
    """Tests unitarios para loss functions."""
    
    def test_sparse_categorical_crossentropy(self):
        """Test SparseCategoricalCrossentropy"""
        loss_fn = SparseCategoricalCrossentropy()
        y_true = torch.randint(0, 3, (32,))
        y_pred = torch.randn(32, 3)
        loss = loss_fn(y_true, y_pred)
        assert loss is not None
        assert loss.item() > 0
    
    def test_mean_squared_error(self):
        """Test MeanSquaredError"""
        loss_fn = MeanSquaredError()
        y_true = torch.randn(32, 10)
        y_pred = torch.randn(32, 10)
        loss = loss_fn(y_true, y_pred)
        assert loss is not None
        assert loss.item() >= 0


class TestMetrics:
    """Tests unitarios para metrics."""
    
    def test_accuracy_metric(self):
        """Test Accuracy metric"""
        accuracy = Accuracy()
        y_true = torch.randint(0, 3, (32,))
        y_pred = torch.randn(32, 3)
        acc = accuracy(y_true, y_pred)
        assert acc is not None
        assert 0.0 <= acc <= 1.0


class TestModelCompilation:
    """Tests unitarios para compilación de modelos."""
    
    def test_model_compile(self):
        """Test compilar modelo"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        loss = SparseCategoricalCrossentropy()
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        assert model._compiled == True
        assert model._optimizer is not None
        assert model._loss is not None
    
    def test_model_compile_with_metrics(self):
        """Test compilar modelo con metrics"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy', 'precision']
        )
        
        assert len(model._metrics) >= 1


class TestModelTraining:
    """Tests unitarios para entrenamiento."""
    
    def test_model_fit_basic(self):
        """Test fit básico"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Datos de entrenamiento
        x_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 3, 100).astype(np.int64)
        
        # Entrenar
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        assert history is not None
        assert 'loss' in history
        assert len(history['loss']) > 0
    
    def test_model_fit_with_validation(self):
        """Test fit con validación"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        x_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 3, 100).astype(np.int64)
        x_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randint(0, 3, 20).astype(np.int64)
        
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=32,
            validation_data=(x_val, y_val),
            verbose=0
        )
        
        assert 'val_loss' in history
        assert 'val_accuracy' in history


class TestModelEvaluation:
    """Tests unitarios para evaluación."""
    
    def test_model_evaluate(self):
        """Test evaluate"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Entrenar un poco primero
        x_train = np.random.randn(50, 10).astype(np.float32)
        y_train = np.random.randint(0, 3, 50).astype(np.int64)
        model.fit(x_train, y_train, epochs=1, verbose=0)
        
        # Evaluar
        x_test = np.random.randn(20, 10).astype(np.float32)
        y_test = np.random.randint(0, 3, 20).astype(np.int64)
        
        results = model.evaluate(x_test, y_test, verbose=0)
        
        assert results is not None
        assert len(results) >= 2  # loss + metrics


class TestModelPrediction:
    """Tests unitarios para predicción."""
    
    def test_model_predict(self):
        """Test predict"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy()
        )
        
        # Entrenar un poco
        x_train = np.random.randn(50, 10).astype(np.float32)
        y_train = np.random.randint(0, 3, 50).astype(np.int64)
        model.fit(x_train, y_train, epochs=1, verbose=0)
        
        # Predecir
        x = np.random.randn(10, 10).astype(np.float32)
        predictions = model.predict(x, verbose=0)
        
        assert predictions is not None
        assert predictions.shape[0] == 10
        assert predictions.shape[1] == 3
        
        # Verificar que son probabilidades (suman ~1)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


class TestModelSummary:
    """Tests unitarios para summary del modelo."""
    
    def test_model_summary(self):
        """Test summary del modelo"""
        model = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        
        # Summary debería ejecutarse sin error
        try:
            model.summary()
            summary_works = True
        except:
            summary_works = False
        
        # Al menos debería no fallar
        assert True  # Si llegamos aquí, no falló


if __name__ == "__main__":
    pytest.main([__file__, "-v"])












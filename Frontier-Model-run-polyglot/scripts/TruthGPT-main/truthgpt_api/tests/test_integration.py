"""
Tests de Integración
====================
Tests que verifican la integración entre diferentes componentes
"""

import pytest
import requests
import numpy as np
import time

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

@pytest.fixture(scope="module")
def server_running():
    """Verifica que el servidor esté corriendo."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    pytest.skip("Servidor no está corriendo. Ejecuta: python start_server.py")


class TestLayerIntegration:
    """Tests de integración entre diferentes layers."""
    
    def test_dense_dropout_integration(self, server_running):
        """Test integración Dense + Dropout"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.3}},
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        model_id = response.json()["model_id"]
        
        # Compilar y entrenar
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        x_train = np.random.randn(50, 128).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 50).astype(np.int64).tolist()
        
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert train_response.status_code == 200
    
    def test_conv_pooling_integration(self, server_running):
        """Test integración Conv2D + Pooling"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "conv2d", "params": {"filters": 64, "kernel_size": 3}},
                    {"type": "averagepooling2d", "params": {"pool_size": 2}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 10}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestOptimizerLossIntegration:
    """Tests de integración entre optimizers y losses."""
    
    def test_adam_with_different_losses(self, server_running):
        """Test Adam con diferentes loss functions"""
        losses = ["sparsecategoricalcrossentropy", "mse", "mae"]
        
        for loss in losses:
            create_response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}}
                    ]
                },
                timeout=TIMEOUT
            )
            model_id = create_response.json()["model_id"]
            
            compile_response = requests.post(
                f"{BASE_URL}/models/{model_id}/compile",
                json={"optimizer": "adam", "loss": loss},
                timeout=TIMEOUT
            )
            assert compile_response.status_code == 200
    
    def test_different_optimizers_with_same_loss(self, server_running):
        """Test diferentes optimizers con el mismo loss"""
        optimizers = ["adam", "sgd", "rmsprop"]
        
        for optimizer in optimizers:
            create_response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}}
                    ]
                },
                timeout=TIMEOUT
            )
            model_id = create_response.json()["model_id"]
            
            compile_response = requests.post(
                f"{BASE_URL}/models/{model_id}/compile",
                json={"optimizer": optimizer, "loss": "mse"},
                timeout=TIMEOUT
            )
            assert compile_response.status_code == 200


class TestDataIntegration:
    """Tests de integración con diferentes tipos de datos."""
    
    def test_different_data_shapes(self, server_running):
        """Test con diferentes formas de datos"""
        shapes = [
            (100, 10),
            (50, 20),
            (200, 5),
            (75, 15)
        ]
        
        for n_samples, n_features in shapes:
            create_response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": n_features, "activation": "relu"}},
                        {"type": "dense", "params": {"units": 3, "activation": "softmax"}}
                    ]
                },
                timeout=TIMEOUT
            )
            model_id = create_response.json()["model_id"]
            
            requests.post(
                f"{BASE_URL}/models/{model_id}/compile",
                json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
                timeout=TIMEOUT
            )
            
            x_train = np.random.randn(n_samples, n_features).astype(np.float32).tolist()
            y_train = np.random.randint(0, 3, n_samples).astype(np.int64).tolist()
            
            train_response = requests.post(
                f"{BASE_URL}/models/{model_id}/train",
                json={
                    "x_train": x_train,
                    "y_train": y_train,
                    "epochs": 1,
                    "verbose": 0
                },
                timeout=TIMEOUT * 2
            )
            assert train_response.status_code == 200
    
    def test_batch_size_variations(self, server_running):
        """Test con diferentes batch sizes"""
        batch_sizes = [1, 8, 16, 32, 64]
        
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(100, 32).astype(np.float32).tolist()
        y_train = np.random.randn(100, 10).astype(np.float32).tolist()
        
        for batch_size in batch_sizes:
            train_response = requests.post(
                f"{BASE_URL}/models/{model_id}/train",
                json={
                    "x_train": x_train,
                    "y_train": y_train,
                    "epochs": 1,
                    "batch_size": batch_size,
                    "verbose": 0
                },
                timeout=TIMEOUT * 2
            )
            # Puede fallar con batch_size muy pequeño o grande, pero debería manejar
            assert train_response.status_code in [200, 400]


class TestMetricsIntegration:
    """Tests de integración con metrics."""
    
    def test_multiple_metrics(self, server_running):
        """Test con múltiples metrics"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy", "precision", "recall"]
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200


class TestConcurrentOperations:
    """Tests de operaciones concurrentes."""
    
    def test_concurrent_model_creation(self, server_running):
        """Test crear múltiples modelos concurrentemente"""
        import concurrent.futures
        
        def create_model(i):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}},
                        {"type": "dense", "params": {"units": 10}}
                    ],
                    "name": f"concurrent-model-{i}"
                },
                timeout=TIMEOUT
            )
            return response.status_code == 200
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_model, i) for i in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert all(results)
    
    def test_concurrent_compile(self, server_running):
        """Test compilar múltiples modelos concurrentemente"""
        # Crear modelos primero
        model_ids = []
        for i in range(3):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}}
                    ]
                },
                timeout=TIMEOUT
            )
            model_ids.append(response.json()["model_id"])
        
        # Compilar concurrentemente
        import concurrent.futures
        
        def compile_model(model_id):
            response = requests.post(
                f"{BASE_URL}/models/{model_id}/compile",
                json={"optimizer": "adam", "loss": "mse"},
                timeout=TIMEOUT
            )
            return response.status_code == 200
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(compile_model, mid) for mid in model_ids]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])












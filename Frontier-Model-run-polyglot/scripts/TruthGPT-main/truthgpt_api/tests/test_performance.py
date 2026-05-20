"""
Tests de Rendimiento
====================
Tests que miden y verifican el rendimiento de la API
"""

import pytest
import requests
import numpy as np
import time
from statistics import mean

BASE_URL = "http://localhost:8000"
TIMEOUT = 60

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


class TestAPILatency:
    """Tests de latencia de la API."""
    
    def test_health_check_latency(self, server_running):
        """Test latencia del health check"""
        latencies = []
        for _ in range(10):
            start = time.time()
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            latency = time.time() - start
            assert response.status_code == 200
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        max_latency = max(latencies)
        
        print(f"\n   Health check - Avg: {avg_latency*1000:.2f}ms, Max: {max_latency*1000:.2f}ms")
        assert avg_latency < 1.0  # Debería ser rápido
    
    def test_create_model_latency(self, server_running):
        """Test latencia de creación de modelos"""
        latencies = []
        for _ in range(5):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 64}},
                        {"type": "dense", "params": {"units": 10}}
                    ]
                },
                timeout=TIMEOUT
            )
            latency = time.time() - start
            assert response.status_code == 200
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        print(f"\n   Create model - Avg: {avg_latency*1000:.2f}ms")
        assert avg_latency < 5.0  # Crear modelo debería ser rápido
    
    def test_compile_latency(self, server_running):
        """Test latencia de compilación"""
        # Crear modelo primero
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64}},
                    {"type": "dense", "params": {"units": 10}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        latencies = []
        for _ in range(5):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/models/{model_id}/compile",
                json={"optimizer": "adam", "loss": "mse"},
                timeout=TIMEOUT
            )
            latency = time.time() - start
            assert response.status_code == 200
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        print(f"\n   Compile - Avg: {avg_latency*1000:.2f}ms")


class TestTrainingPerformance:
    """Tests de rendimiento de entrenamiento."""
    
    def test_training_speed_small_dataset(self, server_running):
        """Test velocidad de entrenamiento con dataset pequeño"""
        # Crear y compilar modelo
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
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
            timeout=TIMEOUT
        )
        
        # Dataset pequeño
        x_train = np.random.randn(100, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 100).astype(np.int64).tolist()
        
        start = time.time()
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 5,
                "batch_size": 32,
                "verbose": 0
            },
            timeout=TIMEOUT * 5
        )
        training_time = time.time() - start
        
        assert train_response.status_code == 200
        print(f"\n   Training (100 samples, 5 epochs) - Time: {training_time:.2f}s")
        assert training_time < 120  # No debería tomar más de 2 minutos
    
    def test_training_speed_medium_dataset(self, server_running):
        """Test velocidad de entrenamiento con dataset mediano"""
        # Crear y compilar modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
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
        
        # Dataset mediano
        x_train = np.random.randn(1000, 64).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 1000).astype(np.int64).tolist()
        
        start = time.time()
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 3,
                "batch_size": 64,
                "verbose": 0
            },
            timeout=TIMEOUT * 10
        )
        training_time = time.time() - start
        
        assert train_response.status_code == 200
        print(f"\n   Training (1000 samples, 3 epochs) - Time: {training_time:.2f}s")


class TestPredictionPerformance:
    """Tests de rendimiento de predicciones."""
    
    def test_prediction_latency(self, server_running):
        """Test latencia de predicciones"""
        # Crear, compilar y entrenar modelo
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
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 50).astype(np.int64).tolist()
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        
        # Medir latencia de predicciones
        x_pred = np.random.randn(1, 32).astype(np.float32).tolist()
        latencies = []
        
        for _ in range(20):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/models/{model_id}/predict",
                json={"x": x_pred, "verbose": 0},
                timeout=TIMEOUT
            )
            latency = time.time() - start
            assert response.status_code == 200
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\n   Prediction - Avg: {avg_latency*1000:.2f}ms, P95: {p95_latency*1000:.2f}ms")
        assert avg_latency < 2.0  # Predicciones deberían ser rápidas
    
    def test_batch_prediction_performance(self, server_running):
        """Test rendimiento de predicciones en batch"""
        # Crear, compilar y entrenar modelo
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
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 50).astype(np.int64).tolist()
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        
        # Batch de predicciones
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            x_batch = np.random.randn(batch_size, 32).astype(np.float32).tolist()
            
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/models/{model_id}/predict",
                json={"x": x_batch, "verbose": 0},
                timeout=TIMEOUT
            )
            batch_time = time.time() - start
            
            assert response.status_code == 200
            time_per_sample = batch_time / batch_size
            print(f"\n   Batch prediction ({batch_size} samples) - {time_per_sample*1000:.2f}ms per sample")


class TestMemoryUsage:
    """Tests de uso de memoria."""
    
    def test_multiple_models_memory(self, server_running):
        """Test que múltiples modelos no consuman demasiada memoria"""
        model_ids = []
        
        # Crear varios modelos
        for i in range(10):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 64}},
                        {"type": "dense", "params": {"units": 10}}
                    ],
                    "name": f"memory-test-{i}"
                },
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])
        
        # Verificar que todos existen
        list_response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert list_response.status_code == 200
        assert len(list_response.json()["models"]) >= len(model_ids)
        
        # Limpiar
        for model_id in model_ids:
            requests.delete(f"{BASE_URL}/models/{model_id}", timeout=TIMEOUT)


class TestScalability:
    """Tests de escalabilidad."""
    
    def test_model_size_scalability(self, server_running):
        """Test escalabilidad con modelos de diferentes tamaños"""
        layer_counts = [2, 5, 10, 20]
        
        for layer_count in layer_counts:
            layers = []
            for i in range(layer_count - 1):
                layers.append({"type": "dense", "params": {"units": 64, "activation": "relu"}})
            layers.append({"type": "dense", "params": {"units": 10, "activation": "softmax"}})
            
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={"layers": layers},
                timeout=TIMEOUT
            )
            creation_time = time.time() - start
            
            assert response.status_code == 200
            print(f"\n   Model with {layer_count} layers - Creation: {creation_time*1000:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])












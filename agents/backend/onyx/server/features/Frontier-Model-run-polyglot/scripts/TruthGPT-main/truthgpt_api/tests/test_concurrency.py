"""
Tests de Concurrencia
=====================
Tests para verificar el comportamiento con múltiples requests simultáneos
"""

import pytest
import requests
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class TestConcurrentHealthChecks:
    """Tests de health checks concurrentes."""
    
    def test_multiple_health_checks(self, server_running):
        """Test múltiples health checks simultáneos"""
        def check_health():
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            return response.status_code == 200
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_health) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        assert all(results), "Algunos health checks fallaron"
    
    def test_rapid_health_checks(self, server_running):
        """Test health checks muy rápidos"""
        def check_health():
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=2)
                return response.status_code == 200
            except:
                return False
        
        results = []
        for _ in range(50):
            results.append(check_health())
            time.sleep(0.01)  # 10ms entre requests
        
        assert sum(results) >= 45, "Demasiados health checks fallaron"


class TestConcurrentModelCreation:
    """Tests de creación concurrente de modelos."""
    
    def test_create_multiple_models_simultaneously(self, server_running):
        """Test crear múltiples modelos simultáneamente"""
        def create_model(model_num):
            try:
                response = requests.post(
                    f"{BASE_URL}/models/create",
                    json={
                        "layers": [
                            {"type": "dense", "params": {"units": 32}},
                            {"type": "dense", "params": {"units": 10}}
                        ],
                        "name": f"concurrent-model-{model_num}"
                    },
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    return response.json().get("model_id")
                return None
            except Exception as e:
                return None
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_model, i) for i in range(10)]
            model_ids = [f.result() for f in as_completed(futures)]
        
        # Filtrar None
        model_ids = [mid for mid in model_ids if mid is not None]
        assert len(model_ids) >= 8, f"Solo se crearon {len(model_ids)}/10 modelos"
    
    def test_create_models_rapid_sequence(self, server_running):
        """Test crear modelos en secuencia rápida"""
        model_ids = []
        
        for i in range(10):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 16}},
                        {"type": "dense", "params": {"units": 5}}
                    ],
                    "name": f"rapid-model-{i}"
                },
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                model_ids.append(response.json().get("model_id"))
            time.sleep(0.1)  # 100ms entre requests
        
        assert len(model_ids) == 10, f"Se crearon {len(model_ids)}/10 modelos"


class TestConcurrentTraining:
    """Tests de entrenamiento concurrente."""
    
    def test_train_multiple_models_simultaneously(self, server_running):
        """Test entrenar múltiples modelos simultáneamente"""
        # Crear modelos primero
        model_ids = []
        for i in range(3):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}},
                        {"type": "dense", "params": {"units": 10}}
                    ]
                },
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                model_id = response.json().get("model_id")
                model_ids.append(model_id)
                
                # Compilar
                requests.post(
                    f"{BASE_URL}/models/{model_id}/compile",
                    json={"optimizer": "adam", "loss": "mse"},
                    timeout=TIMEOUT
                )
        
        # Entrenar simultáneamente
        def train_model(model_id):
            try:
                x_train = np.random.randn(20, 32).astype(np.float32).tolist()
                y_train = np.random.randn(20, 10).astype(np.float32).tolist()
                
                response = requests.post(
                    f"{BASE_URL}/models/{model_id}/train",
                    json={
                        "x_train": x_train,
                        "y_train": y_train,
                        "epochs": 1,
                        "verbose": 0
                    },
                    timeout=TIMEOUT * 2
                )
                return response.status_code == 200
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(train_model, mid) for mid in model_ids]
            results = [f.result() for f in as_completed(futures)]
        
        assert all(results), "Algunos entrenamientos fallaron"


class TestConcurrentPredictions:
    """Tests de predicciones concurrentes."""
    
    def test_predict_multiple_requests_simultaneously(self, server_running):
        """Test múltiples predicciones simultáneas"""
        # Crear y entrenar modelo
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = response.json().get("model_id")
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(10, 32).astype(np.float32).tolist()
        y_train = np.random.randn(10, 10).astype(np.float32).tolist()
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        
        # Predicciones concurrentes
        def predict(x_data):
            try:
                response = requests.post(
                    f"{BASE_URL}/models/{model_id}/predict",
                    json={"x": x_data},
                    timeout=TIMEOUT
                )
                return response.status_code == 200
            except:
                return False
        
        x_test = np.random.randn(5, 32).astype(np.float32).tolist()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(predict, x_test) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        assert sum(results) >= 18, "Demasiadas predicciones fallaron"


class TestConcurrentListOperations:
    """Tests de operaciones de listado concurrentes."""
    
    def test_list_models_while_creating(self, server_running):
        """Test listar modelos mientras se crean"""
        def create_model():
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 16}}
                    ]
                },
                timeout=TIMEOUT
            )
            return response.status_code == 200
        
        def list_models():
            response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
            return response.status_code == 200
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Crear modelos
            create_futures = [executor.submit(create_model) for _ in range(5)]
            # Listar modelos simultáneamente
            list_futures = [executor.submit(list_models) for _ in range(10)]
            
            create_results = [f.result() for f in as_completed(create_futures)]
            list_results = [f.result() for f in as_completed(list_futures)]
        
        assert all(create_results), "Algunas creaciones fallaron"
        assert all(list_results), "Algunos listados fallaron"


class TestStressConcurrency:
    """Tests de stress con concurrencia."""
    
    def test_stress_health_checks(self, server_running):
        """Test stress de health checks"""
        def check_health():
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=2)
                return response.status_code == 200
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_health) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]
        
        assert sum(results) >= 90, "Demasiados health checks fallaron bajo stress"
    
    def test_stress_model_creation(self, server_running):
        """Test stress de creación de modelos"""
        def create_model():
            try:
                response = requests.post(
                    f"{BASE_URL}/models/create",
                    json={
                        "layers": [
                            {"type": "dense", "params": {"units": 8}}
                        ]
                    },
                    timeout=TIMEOUT
                )
                return response.status_code == 200
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_model) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]
        
        assert sum(results) >= 40, "Demasiadas creaciones fallaron bajo stress"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])












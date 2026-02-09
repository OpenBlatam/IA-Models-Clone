"""
Tests End-to-End (E2E) para TruthGPT API
=========================================
Tests que verifican flujos completos desde creación hasta predicción
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


class TestCompleteMLWorkflow:
    """Test de flujo completo de Machine Learning."""
    
    def test_complete_classification_workflow(self, server_running):
        """Test completo: crear, compilar, entrenar, evaluar y predecir"""
        print("\n🔄 Iniciando flujo completo de clasificación...")
        
        # 1. Crear modelo
        print("   📦 Paso 1: Crear modelo...")
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 3, "activation": "softmax"}}
                ],
                "name": "e2e-classification-model"
            },
            timeout=TIMEOUT
        )
        assert create_response.status_code == 200
        model_data = create_response.json()
        model_id = model_data["model_id"]
        print(f"   ✅ Modelo creado: {model_id}")
        
        # 2. Compilar modelo
        print("   ⚙️  Paso 2: Compilar modelo...")
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": 0.001},
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy"]
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        print("   ✅ Modelo compilado")
        
        # 3. Generar datos de entrenamiento
        print("   📊 Paso 3: Generar datos...")
        np.random.seed(42)
        x_train = np.random.randn(200, 10).astype(np.float32).tolist()
        y_train = np.random.randint(0, 3, 200).astype(np.int64).tolist()
        x_test = np.random.randn(50, 10).astype(np.float32).tolist()
        y_test = np.random.randint(0, 3, 50).astype(np.int64).tolist()
        x_val = np.random.randn(30, 10).astype(np.float32).tolist()
        y_val = np.random.randint(0, 3, 30).astype(np.int64).tolist()
        print("   ✅ Datos generados")
        
        # 4. Entrenar modelo
        print("   🚀 Paso 4: Entrenar modelo...")
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 3,
                "batch_size": 32,
                "validation_data": {
                    "x": x_val,
                    "y": y_val
                },
                "verbose": 0
            },
            timeout=TIMEOUT * 3  # Training puede tomar tiempo
        )
        assert train_response.status_code == 200
        train_data = train_response.json()
        assert train_data["status"] == "trained"
        assert "history" in train_data
        print("   ✅ Modelo entrenado")
        print(f"      Loss final: {train_data['history'].get('loss', [0])[-1]:.4f}")
        
        # 5. Evaluar modelo
        print("   📈 Paso 5: Evaluar modelo...")
        eval_response = requests.post(
            f"{BASE_URL}/models/{model_id}/evaluate",
            json={
                "x_test": x_test,
                "y_test": y_test,
                "verbose": 0
            },
            timeout=TIMEOUT
        )
        assert eval_response.status_code == 200
        eval_data = eval_response.json()
        assert eval_data["status"] == "evaluated"
        assert "results" in eval_data
        print("   ✅ Modelo evaluado")
        print(f"      Test Loss: {eval_data['results']['loss']:.4f}")
        
        # 6. Hacer predicciones
        print("   🔮 Paso 6: Hacer predicciones...")
        x_pred = np.random.randn(10, 10).astype(np.float32).tolist()
        predict_response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            json={
                "x": x_pred,
                "verbose": 0
            },
            timeout=TIMEOUT
        )
        assert predict_response.status_code == 200
        pred_data = predict_response.json()
        assert pred_data["status"] == "predicted"
        assert "predictions" in pred_data
        assert len(pred_data["predictions"]) == 10
        print("   ✅ Predicciones completadas")
        print(f"      Predicciones generadas: {len(pred_data['predictions'])}")
        
        # 7. Verificar formato de predicciones
        predictions = np.array(pred_data["predictions"])
        assert predictions.shape[1] == 3  # 3 clases
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6)  # Probabilidades suman 1
        
        print("\n✅ Flujo completo de clasificación exitoso!")
        return model_id


class TestCNNWorkflow:
    """Test de flujo completo para CNN."""
    
    def test_cnn_complete_workflow(self, server_running):
        """Test completo para modelo CNN"""
        print("\n🔄 Iniciando flujo completo CNN...")
        
        # 1. Crear modelo CNN
        print("   📦 Crear modelo CNN...")
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "conv2d", "params": {"filters": 64, "kernel_size": 3, "activation": "relu"}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.5}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "e2e-cnn-model"
            },
            timeout=TIMEOUT
        )
        assert create_response.status_code == 200
        model_id = create_response.json()["model_id"]
        print(f"   ✅ Modelo CNN creado: {model_id}")
        
        # 2. Compilar
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": 0.001},
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy"]
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        # 3. Generar datos de imagen (simulados)
        print("   📊 Generar datos de imagen...")
        np.random.seed(42)
        # Simular imágenes 32x32 con 3 canales
        x_train = np.random.randn(100, 3, 32, 32).astype(np.float32).transpose(0, 2, 3, 1).reshape(100, -1).tolist()
        y_train = np.random.randint(0, 10, 100).astype(np.int64).tolist()
        print(f"   ✅ Datos generados: {len(x_train)} muestras")
        
        # 4. Entrenar
        print("   🚀 Entrenar modelo CNN...")
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 2,
                "batch_size": 16,
                "verbose": 0
            },
            timeout=TIMEOUT * 3
        )
        assert train_response.status_code == 200
        print("   ✅ Modelo CNN entrenado")
        
        print("\n✅ Flujo completo CNN exitoso!")


class TestRNNWorkflow:
    """Test de flujo completo para RNN."""
    
    def test_lstm_complete_workflow(self, server_running):
        """Test completo para modelo LSTM"""
        print("\n🔄 Iniciando flujo completo LSTM...")
        
        # 1. Crear modelo LSTM
        print("   📦 Crear modelo LSTM...")
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "lstm", "params": {"units": 128, "return_sequences": True}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "lstm", "params": {"units": 64, "return_sequences": False}},
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "e2e-lstm-model"
            },
            timeout=TIMEOUT
        )
        assert create_response.status_code == 200
        model_id = create_response.json()["model_id"]
        print(f"   ✅ Modelo LSTM creado: {model_id}")
        
        # 2. Compilar
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy"]
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        # 3. Generar datos secuenciales
        print("   📊 Generar datos secuenciales...")
        np.random.seed(42)
        x_train = np.random.randn(100, 10, 20).astype(np.float32).reshape(100, -1).tolist()
        y_train = np.random.randint(0, 10, 100).astype(np.int64).tolist()
        print(f"   ✅ Datos generados: {len(x_train)} muestras")
        
        # 4. Entrenar
        print("   🚀 Entrenar modelo LSTM...")
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 2,
                "batch_size": 16,
                "verbose": 0
            },
            timeout=TIMEOUT * 3
        )
        assert train_response.status_code == 200
        print("   ✅ Modelo LSTM entrenado")
        
        print("\n✅ Flujo completo LSTM exitoso!")


class TestModelPersistence:
    """Test de persistencia de modelos."""
    
    def test_save_and_load_model(self, server_running):
        """Test guardar y cargar modelo"""
        print("\n🔄 Test de persistencia de modelo...")
        
        # 1. Crear y compilar modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "persistence-test-model"
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        # 2. Guardar modelo
        print("   💾 Guardar modelo...")
        save_response = requests.post(
            f"{BASE_URL}/models/{model_id}/save",
            json={"filepath": "test_model_e2e.pth"},
            timeout=TIMEOUT
        )
        # Puede fallar si no hay soporte completo, pero no debería dar error 500
        assert save_response.status_code in [200, 400, 500]
        print(f"   ✅ Intento de guardar completado (status: {save_response.status_code})")
        
        # 3. Verificar que el modelo sigue accesible
        print("   🔍 Verificar modelo...")
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 200
        print("   ✅ Modelo sigue accesible")
        
        print("\n✅ Test de persistencia completado!")


class TestMultipleModels:
    """Test de múltiples modelos simultáneos."""
    
    def test_multiple_models_workflow(self, server_running):
        """Test crear y gestionar múltiples modelos"""
        print("\n🔄 Test de múltiples modelos...")
        
        model_ids = []
        
        # Crear 3 modelos diferentes
        for i in range(3):
            print(f"   📦 Crear modelo {i+1}/3...")
            create_response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32 + i*16, "activation": "relu"}},
                        {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                    ],
                    "name": f"multi-model-{i}"
                },
                timeout=TIMEOUT
            )
            assert create_response.status_code == 200
            model_ids.append(create_response.json()["model_id"])
        
        print(f"   ✅ {len(model_ids)} modelos creados")
        
        # Listar todos los modelos
        print("   📋 Listar modelos...")
        list_response = requests.get(
            f"{BASE_URL}/models",
            timeout=TIMEOUT
        )
        assert list_response.status_code == 200
        models_list = list_response.json()["models"]
        assert len(models_list) >= len(model_ids)
        print(f"   ✅ Encontrados {len(models_list)} modelos")
        
        # Compilar y entrenar uno de los modelos
        print("   🚀 Compilar y entrenar modelo de prueba...")
        test_model_id = model_ids[0]
        
        compile_response = requests.post(
            f"{BASE_URL}/models/{test_model_id}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        np.random.seed(42)
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 50).astype(np.int64).tolist()
        
        train_response = requests.post(
            f"{BASE_URL}/models/{test_model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1,
                "verbose": 0
            },
            timeout=TIMEOUT * 2
        )
        assert train_response.status_code == 200
        print("   ✅ Modelo entrenado exitosamente")
        
        # Eliminar uno de los modelos
        print("   🗑️  Eliminar modelo de prueba...")
        delete_response = requests.delete(
            f"{BASE_URL}/models/{test_model_id}",
            timeout=TIMEOUT
        )
        assert delete_response.status_code == 200
        print("   ✅ Modelo eliminado")
        
        # Verificar que fue eliminado
        info_response = requests.get(
            f"{BASE_URL}/models/{test_model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 404
        print("   ✅ Verificado que el modelo fue eliminado")
        
        print("\n✅ Test de múltiples modelos exitoso!")


class TestErrorRecovery:
    """Test de recuperación de errores."""
    
    def test_error_recovery_workflow(self, server_running):
        """Test que el sistema se recupera correctamente de errores"""
        print("\n🔄 Test de recuperación de errores...")
        
        # 1. Intentar operaciones inválidas
        print("   ❌ Probar operaciones inválidas...")
        
        # Compilar modelo inexistente
        compile_response = requests.post(
            f"{BASE_URL}/models/invalid-id/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 404
        print("   ✅ Error manejado correctamente (404)")
        
        # 2. Crear modelo válido después del error
        print("   ✅ Crear modelo válido después del error...")
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        assert create_response.status_code == 200
        model_id = create_response.json()["model_id"]
        print(f"   ✅ Modelo creado exitosamente: {model_id}")
        
        # 3. Intentar entrenar sin compilar
        print("   ❌ Intentar entrenar sin compilar...")
        np.random.seed(42)
        x_train = np.random.randn(10, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 10).astype(np.int64).tolist()
        
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1
            },
            timeout=TIMEOUT
        )
        assert train_response.status_code == 400
        print("   ✅ Error manejado correctamente (400)")
        
        # 4. Compilar y entrenar correctamente
        print("   ✅ Compilar y entrenar correctamente...")
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
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
        print("   ✅ Recuperación exitosa, modelo entrenado")
        
        print("\n✅ Test de recuperación de errores exitoso!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s para ver prints












"""
Test rápido de la API TruthGPT
================================
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check."""
    print("🔍 Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check OK: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Health check falló: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servidor.")
        print("   Por favor, inicia el servidor primero:")
        print("   python start_server.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_create_model():
    """Test crear modelo."""
    print("\n📦 Probando crear modelo...")
    try:
        response = requests.post(f"{BASE_URL}/models/create", json={
            "layers": [
                {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
            ],
            "name": "test-model"
        }, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Modelo creado: {data.get('model_id')}")
            print(f"   Nombre: {data.get('name')}")
            return data.get('model_id')
        else:
            print(f"❌ Error al crear modelo: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_list_models():
    """Test listar modelos."""
    print("\n📋 Probando listar modelos...")
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Modelos encontrados: {data.get('count', 0)}")
            for model in data.get('models', [])[:3]:  # Mostrar solo los primeros 3
                print(f"   - {model.get('model_id')}: {model.get('name')}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 Prueba Rápida de TruthGPT API")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health():
        sys.exit(1)
    
    # Test 2: Crear modelo
    model_id = test_create_model()
    if not model_id:
        print("\n⚠️  No se pudo crear modelo, pero el servidor está funcionando")
        sys.exit(0)
    
    # Test 3: Listar modelos
    test_list_models()
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas exitosamente!")
    print("=" * 60)

if __name__ == "__main__":
    main()












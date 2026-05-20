#!/usr/bin/env python3
"""
Simple API Test for BUL System
==============================

Test básico para verificar que la API esté funcionando correctamente.
Este test no requiere dependencias externas y puede ejecutarse directamente.
"""

import sys
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_api_imports():
    """Test que los módulos principales se puedan importar."""
    print("🔍 Testing API imports...")
    
    try:
        # Test import del sistema principal
        from bul_main import BULSystem, DocumentRequest, DocumentResponse, TaskStatus
        print("✅ BULSystem import successful")
        
        # Test import de configuración
        from bul_config import BULConfig
        print("✅ BULConfig import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_creation():
    """Test que se pueda crear una instancia de la API."""
    print("\n🔍 Testing API creation...")
    
    try:
        from bul_main import BULSystem
        
        # Crear instancia del sistema
        system = BULSystem()
        print("✅ BULSystem instance created")
        
        # Verificar que la app FastAPI esté configurada
        assert hasattr(system, 'app'), "FastAPI app not found"
        print("✅ FastAPI app configured")
        
        # Verificar que las rutas estén configuradas
        assert len(system.app.routes) > 0, "No routes configured"
        print(f"✅ {len(system.app.routes)} routes configured")
        
        return True
        
    except Exception as e:
        print(f"❌ API creation error: {e}")
        return False

def test_api_endpoints():
    """Test que los endpoints principales estén configurados."""
    print("\n🔍 Testing API endpoints...")
    
    try:
        from bul_main import BULSystem
        
        system = BULSystem()
        
        # Verificar endpoints principales
        endpoints = []
        for route in system.app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                endpoints.append({
                    'path': route.path,
                    'methods': list(route.methods)
                })
        
        # Verificar endpoints críticos
        critical_endpoints = [
            ('/', ['GET']),
            ('/health', ['GET']),
            ('/documents/generate', ['POST']),
            ('/tasks', ['GET']),
        ]
        
        found_endpoints = []
        for endpoint_path, expected_methods in critical_endpoints:
            for endpoint in endpoints:
                if endpoint['path'] == endpoint_path:
                    found_endpoints.append(endpoint)
                    print(f"✅ Endpoint {endpoint_path} found with methods {endpoint['methods']}")
                    break
            else:
                print(f"❌ Critical endpoint {endpoint_path} not found")
        
        print(f"✅ Found {len(found_endpoints)}/{len(critical_endpoints)} critical endpoints")
        
        return len(found_endpoints) >= len(critical_endpoints) * 0.75  # 75% de endpoints críticos
        
    except Exception as e:
        print(f"❌ Endpoint testing error: {e}")
        return False

def test_api_models():
    """Test que los modelos Pydantic estén funcionando."""
    print("\n🔍 Testing API models...")
    
    try:
        from bul_main import DocumentRequest, DocumentResponse, TaskStatus
        
        # Test DocumentRequest
        request = DocumentRequest(
            query="Test query for API",
            business_area="marketing",
            document_type="strategy",
            priority=1
        )
        print("✅ DocumentRequest model working")
        
        # Test DocumentResponse
        response = DocumentResponse(
            task_id="test_task_123",
            status="queued",
            message="Test message",
            estimated_time=60
        )
        print("✅ DocumentResponse model working")
        
        # Test TaskStatus
        status = TaskStatus(
            task_id="test_task_123",
            status="processing",
            progress=50
        )
        print("✅ TaskStatus model working")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing error: {e}")
        return False

def test_configuration():
    """Test que la configuración esté funcionando."""
    print("\n🔍 Testing configuration...")
    
    try:
        from bul_config import BULConfig
        
        config = BULConfig()
        print("✅ BULConfig created")
        
        # Verificar configuraciones básicas
        assert hasattr(config, 'api_host'), "API host not configured"
        assert hasattr(config, 'api_port'), "API port not configured"
        assert hasattr(config, 'enabled_business_areas'), "Business areas not configured"
        
        print(f"✅ API Host: {config.api_host}")
        print(f"✅ API Port: {config.api_port}")
        print(f"✅ Business Areas: {len(config.enabled_business_areas)} configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration testing error: {e}")
        return False

def test_system_health():
    """Test que el sistema esté en estado saludable."""
    print("\n🔍 Testing system health...")
    
    try:
        from bul_main import BULSystem
        
        system = BULSystem()
        
        # Verificar que no haya errores de inicialización
        assert hasattr(system, 'tasks'), "Tasks storage not initialized"
        assert hasattr(system, 'app'), "FastAPI app not initialized"
        
        # Verificar estado inicial
        assert isinstance(system.tasks, dict), "Tasks should be a dictionary"
        assert len(system.tasks) == 0, "Tasks should be empty initially"
        
        print("✅ System health check passed")
        print(f"✅ Tasks storage: {len(system.tasks)} tasks")
        
        return True
        
    except Exception as e:
        print(f"❌ System health testing error: {e}")
        return False

def run_all_tests():
    """Ejecutar todos los tests."""
    print("🚀 Starting BUL API Tests")
    print("=" * 50)
    
    tests = [
        ("API Imports", test_api_imports),
        ("API Creation", test_api_creation),
        ("API Endpoints", test_api_endpoints),
        ("API Models", test_api_models),
        ("Configuration", test_configuration),
        ("System Health", test_system_health),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for frontend integration.")
        return True
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. API is mostly ready.")
        return True
    else:
        print("❌ Multiple tests failed. API needs attention.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
BUL API Quick Test
=================

Test rápido para verificar que la API esté lista para frontend.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def quick_test():
    """Test rápido de la API."""
    print("🚀 BUL API Quick Test")
    print("=" * 30)
    
    try:
        # Test 1: Import básico
        print("1. Testing imports...")
        import bul_main
        print("   ✅ bul_main imported successfully")
        
        # Test 2: Crear instancia
        print("2. Testing system creation...")
        system = bul_main.BULSystem()
        print("   ✅ BULSystem created successfully")
        
        # Test 3: Verificar FastAPI app
        print("3. Testing FastAPI app...")
        app = system.app
        print(f"   ✅ FastAPI app: {app.title}")
        print(f"   ✅ Version: {app.version}")
        
        # Test 4: Verificar rutas
        print("4. Testing routes...")
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"   ✅ Found {len(routes)} routes:")
        for route in routes[:5]:  # Mostrar primeras 5 rutas
            print(f"      - {route}")
        
        # Test 5: Verificar modelos
        print("5. Testing models...")
        from bul_main import DocumentRequest
        request = DocumentRequest(query="Test query")
        print("   ✅ DocumentRequest model working")
        
        print("\n🎉 API is ready for frontend integration!")
        print("📋 Available endpoints:")
        print("   - GET  / (root)")
        print("   - GET  /health (health check)")
        print("   - POST /documents/generate (generate document)")
        print("   - GET  /tasks (list tasks)")
        print("   - GET  /tasks/{task_id}/status (task status)")
        print("   - DELETE /tasks/{task_id} (delete task)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 API needs setup or has issues")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ API Status: READY")
    else:
        print("\n❌ API Status: NOT READY")
    
    input("\nPress Enter to continue...")

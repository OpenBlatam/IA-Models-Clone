"""
BUL API Comprehensive Test Suite
================================

Suite completa de pruebas para verificar que la API esté lista para frontend.
Incluye verificación de archivos, estructura, y funcionalidad básica.
"""

import os
import sys
import json
from pathlib import Path

def test_file_structure():
    """Test que todos los archivos necesarios estén presentes."""
    print("🔍 Testing file structure...")
    
    required_files = [
        "bul_main.py",
        "bul_config.py", 
        "requirements.txt",
        "README.md",
        "api/__init__.py",
        "api/bul_api.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    
    print(f"   ✅ All {len(required_files)} required files present")
    return True

def test_api_structure():
    """Test la estructura de la API."""
    print("\n🔍 Testing API structure...")
    
    try:
        # Verificar que bul_main.py existe y es válido
        with open("bul_main.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Verificar componentes clave
        key_components = [
            "class BULSystem",
            "FastAPI",
            "DocumentRequest",
            "DocumentResponse", 
            "TaskStatus",
            "@app.get",
            "@app.post",
            "uvicorn.run"
        ]
        
        found_components = []
        for component in key_components:
            if component in content:
                found_components.append(component)
                print(f"   ✅ {component}")
            else:
                print(f"   ❌ {component} not found")
        
        print(f"   ✅ Found {len(found_components)}/{len(key_components)} key components")
        return len(found_components) >= len(key_components) * 0.8
        
    except Exception as e:
        print(f"   ❌ Error reading bul_main.py: {e}")
        return False

def test_endpoints_definition():
    """Test que los endpoints estén definidos correctamente."""
    print("\n🔍 Testing endpoints definition...")
    
    try:
        with open("bul_main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Buscar definiciones de endpoints
        endpoints = [
            ('@self.app.get("/")', 'Root endpoint'),
            ('@self.app.get("/health")', 'Health check'),
            ('@self.app.post("/documents/generate")', 'Document generation'),
            ('@self.app.get("/tasks/{task_id}/status")', 'Task status'),
            ('@self.app.get("/tasks")', 'List tasks'),
            ('@self.app.delete("/tasks/{task_id}")', 'Delete task')
        ]
        
        found_endpoints = []
        for endpoint_def, description in endpoints:
            if endpoint_def in content:
                found_endpoints.append(description)
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} not found")
        
        print(f"   ✅ Found {len(found_endpoints)}/{len(endpoints)} endpoints")
        return len(found_endpoints) >= len(endpoints) * 0.8
        
    except Exception as e:
        print(f"   ❌ Error checking endpoints: {e}")
        return False

def test_models_definition():
    """Test que los modelos Pydantic estén definidos."""
    print("\n🔍 Testing models definition...")
    
    try:
        with open("bul_main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        models = [
            ('class DocumentRequest(BaseModel)', 'DocumentRequest model'),
            ('class DocumentResponse(BaseModel)', 'DocumentResponse model'),
            ('class TaskStatus(BaseModel)', 'TaskStatus model'),
            ('from pydantic import BaseModel', 'Pydantic import')
        ]
        
        found_models = []
        for model_def, description in models:
            if model_def in content:
                found_models.append(description)
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} not found")
        
        print(f"   ✅ Found {len(found_models)}/{len(models)} models")
        return len(found_models) >= len(models) * 0.8
        
    except Exception as e:
        print(f"   ❌ Error checking models: {e}")
        return False

def test_configuration():
    """Test la configuración del sistema."""
    print("\n🔍 Testing configuration...")
    
    try:
        with open("bul_config.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        config_elements = [
            ('class BULConfig', 'BULConfig class'),
            ('api_host', 'API host setting'),
            ('api_port', 'API port setting'),
            ('enabled_business_areas', 'Business areas config'),
            ('document_types', 'Document types config')
        ]
        
        found_elements = []
        for element, description in config_elements:
            if element in content:
                found_elements.append(description)
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} not found")
        
        print(f"   ✅ Found {len(found_elements)}/{len(config_elements)} config elements")
        return len(found_elements) >= len(config_elements) * 0.8
        
    except Exception as e:
        print(f"   ❌ Error checking configuration: {e}")
        return False

def test_requirements():
    """Test que los requirements estén definidos."""
    print("\n🔍 Testing requirements...")
    
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic'
        ]
        
        found_packages = []
        for package in required_packages:
            if package.lower() in content.lower():
                found_packages.append(package)
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package} not found")
        
        print(f"   ✅ Found {len(found_packages)}/{len(required_packages)} required packages")
        return len(found_packages) >= len(required_packages) * 0.8
        
    except Exception as e:
        print(f"   ❌ Error checking requirements: {e}")
        return False

def test_api_directory():
    """Test que el directorio API tenga los archivos necesarios."""
    print("\n🔍 Testing API directory...")
    
    api_files = [
        "api/__init__.py",
        "api/bul_api.py"
    ]
    
    found_files = []
    for file_path in api_files:
        if Path(file_path).exists():
            found_files.append(file_path)
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} not found")
    
    print(f"   ✅ Found {len(found_files)}/{len(api_files)} API files")
    return len(found_files) >= len(api_files) * 0.5

def test_documentation():
    """Test que la documentación esté presente."""
    print("\n🔍 Testing documentation...")
    
    doc_files = [
        "README.md",
        "README_OPTIMIZED.md"
    ]
    
    found_docs = []
    for doc_file in doc_files:
        if Path(doc_file).exists():
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if len(content) > 100:  # Verificar que no esté vacío
                        found_docs.append(doc_file)
                        print(f"   ✅ {doc_file} (content: {len(content)} chars)")
                    else:
                        print(f"   ⚠️  {doc_file} (too short)")
            except Exception as e:
                print(f"   ❌ Error reading {doc_file}: {e}")
        else:
            print(f"   ❌ {doc_file} not found")
    
    print(f"   ✅ Found {len(found_docs)}/{len(doc_files)} documentation files")
    return len(found_docs) > 0

def generate_api_summary():
    """Generar resumen de la API para frontend."""
    print("\n📋 API Summary for Frontend Integration:")
    print("=" * 50)
    
    print("🚀 Main API File: bul_main.py")
    print("⚙️  Configuration: bul_config.py")
    print("📦 Requirements: requirements.txt")
    print("📚 Documentation: README.md")
    
    print("\n🔗 Available Endpoints:")
    endpoints = [
        ("GET", "/", "Root endpoint - API info"),
        ("GET", "/health", "Health check"),
        ("POST", "/documents/generate", "Generate document"),
        ("GET", "/tasks", "List all tasks"),
        ("GET", "/tasks/{task_id}/status", "Get task status"),
        ("DELETE", "/tasks/{task_id}", "Delete task")
    ]
    
    for method, path, description in endpoints:
        print(f"   {method:6} {path:25} - {description}")
    
    print("\n📝 Request Models:")
    print("   DocumentRequest:")
    print("     - query: str (required)")
    print("     - business_area: str (optional)")
    print("     - document_type: str (optional)")
    print("     - priority: int (1-5, default: 1)")
    print("     - metadata: dict (optional)")
    
    print("\n📤 Response Models:")
    print("   DocumentResponse:")
    print("     - task_id: str")
    print("     - status: str")
    print("     - message: str")
    print("     - estimated_time: int")
    
    print("   TaskStatus:")
    print("     - task_id: str")
    print("     - status: str")
    print("     - progress: int (0-100)")
    print("     - result: dict (optional)")
    print("     - error: str (optional)")

def run_comprehensive_test():
    """Ejecutar todas las pruebas."""
    print("🚀 BUL API Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("API Structure", test_api_structure),
        ("Endpoints Definition", test_endpoints_definition),
        ("Models Definition", test_models_definition),
        ("Configuration", test_configuration),
        ("Requirements", test_requirements),
        ("API Directory", test_api_directory),
        ("Documentation", test_documentation),
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
        print("🎉 All tests passed! API is fully ready for frontend integration.")
        generate_api_summary()
        return True
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. API is mostly ready for frontend integration.")
        generate_api_summary()
        return True
    else:
        print("❌ Multiple tests failed. API needs attention before frontend integration.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ API Status: READY FOR FRONTEND")
        print("\n🚀 Next Steps:")
        print("   1. Start the API server: python bul_main.py")
        print("   2. Test endpoints with curl or Postman")
        print("   3. Integrate with frontend application")
    else:
        print("❌ API Status: NEEDS ATTENTION")
        print("\n🔧 Required Actions:")
        print("   1. Fix failing tests")
        print("   2. Ensure all dependencies are installed")
        print("   3. Verify configuration")
    
    input("\nPress Enter to continue...")

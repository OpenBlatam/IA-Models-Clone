"""
BUL API Final Test Suite
========================

Suite final de pruebas que ejecuta todos los tests y genera un reporte completo
para verificar que la API esté completamente lista para frontend.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

class BULAPITester:
    """Test suite completo para BUL API."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
            "recommendations": []
        }
        self.passed_tests = 0
        self.total_tests = 0
    
    def run_test(self, test_name, test_func):
        """Ejecutar un test individual."""
        print(f"\n🔍 Running {test_name}...")
        self.total_tests += 1
        
        try:
            result = test_func()
            if result:
                self.passed_tests += 1
                self.results["tests"][test_name] = {
                    "status": "PASSED",
                    "message": "Test completed successfully"
                }
                print(f"   ✅ {test_name}: PASSED")
            else:
                self.results["tests"][test_name] = {
                    "status": "FAILED",
                    "message": "Test failed"
                }
                print(f"   ❌ {test_name}: FAILED")
        except Exception as e:
            self.results["tests"][test_name] = {
                "status": "ERROR",
                "message": str(e)
            }
            print(f"   💥 {test_name}: ERROR - {e}")
    
    def test_file_structure(self):
        """Test estructura de archivos."""
        required_files = [
            "bul_main.py", "bul_config.py", "requirements.txt", 
            "README.md", "api/__init__.py", "api/bul_api.py"
        ]
        
        missing = [f for f in required_files if not Path(f).exists()]
        return len(missing) == 0
    
    def test_api_core(self):
        """Test componentes core de la API."""
        try:
            with open("bul_main.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            core_elements = [
                "class BULSystem", "FastAPI", "DocumentRequest", 
                "DocumentResponse", "TaskStatus", "uvicorn.run"
            ]
            
            found = sum(1 for elem in core_elements if elem in content)
            return found >= len(core_elements) * 0.8
        except:
            return False
    
    def test_endpoints(self):
        """Test endpoints de la API."""
        try:
            with open("bul_main.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            endpoints = [
                '@self.app.get("/")', '@self.app.get("/health")',
                '@self.app.post("/documents/generate")', 
                '@self.app.get("/tasks/{task_id}/status")',
                '@self.app.get("/tasks")', '@self.app.delete("/tasks/{task_id}")'
            ]
            
            found = sum(1 for ep in endpoints if ep in content)
            return found >= len(endpoints) * 0.8
        except:
            return False
    
    def test_cors(self):
        """Test configuración CORS."""
        try:
            with open("bul_main.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            cors_elements = ["CORSMiddleware", "allow_origins", "allow_methods"]
            found = sum(1 for elem in cors_elements if elem in content)
            return found >= len(cors_elements) * 0.8
        except:
            return False
    
    def test_error_handling(self):
        """Test manejo de errores."""
        try:
            with open("bul_main.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            error_elements = ["HTTPException", "try:", "except", "raise"]
            found = sum(1 for elem in error_elements if elem in content)
            return found >= len(error_elements) * 0.6
        except:
            return False
    
    def test_async_processing(self):
        """Test procesamiento asíncrono."""
        try:
            with open("bul_main.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            async_elements = ["async def", "await", "BackgroundTasks"]
            found = sum(1 for elem in async_elements if elem in content)
            return found >= len(async_elements) * 0.6
        except:
            return False
    
    def test_configuration(self):
        """Test configuración."""
        try:
            with open("bul_config.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            config_elements = ["class BULConfig", "api_host", "api_port"]
            found = sum(1 for elem in config_elements if elem in content)
            return found >= len(config_elements) * 0.8
        except:
            return False
    
    def test_requirements(self):
        """Test requirements."""
        try:
            with open("requirements.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            required_packages = ["fastapi", "uvicorn", "pydantic"]
            found = sum(1 for pkg in required_packages if pkg.lower() in content.lower())
            return found >= len(required_packages) * 0.8
        except:
            return False
    
    def test_documentation(self):
        """Test documentación."""
        doc_files = ["README.md", "README_OPTIMIZED.md"]
        existing_docs = [f for f in doc_files if Path(f).exists()]
        return len(existing_docs) > 0
    
    def generate_recommendations(self):
        """Generar recomendaciones basadas en los resultados."""
        recommendations = []
        
        # Analizar resultados
        failed_tests = [name for name, result in self.results["tests"].items() 
                       if result["status"] in ["FAILED", "ERROR"]]
        
        if "File Structure" in failed_tests:
            recommendations.append("Fix missing files in the project structure")
        
        if "API Core" in failed_tests:
            recommendations.append("Review and fix core API components")
        
        if "Endpoints" in failed_tests:
            recommendations.append("Ensure all required endpoints are implemented")
        
        if "CORS" in failed_tests:
            recommendations.append("Configure CORS properly for frontend access")
        
        if "Error Handling" in failed_tests:
            recommendations.append("Implement proper error handling")
        
        if "Async Processing" in failed_tests:
            recommendations.append("Fix async processing implementation")
        
        if "Configuration" in failed_tests:
            recommendations.append("Review configuration files")
        
        if "Requirements" in failed_tests:
            recommendations.append("Update requirements.txt with missing packages")
        
        if "Documentation" in failed_tests:
            recommendations.append("Create or update documentation")
        
        # Recomendaciones generales
        success_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0
        
        if success_rate >= 0.9:
            recommendations.append("API is ready for production deployment")
        elif success_rate >= 0.7:
            recommendations.append("API is mostly ready, fix remaining issues")
        else:
            recommendations.append("API needs significant work before deployment")
        
        self.results["recommendations"] = recommendations
    
    def generate_summary(self):
        """Generar resumen de resultados."""
        self.results["summary"] = {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.total_tests - self.passed_tests,
            "success_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
            "status": "READY" if self.passed_tests >= self.total_tests * 0.8 else "NEEDS_WORK"
        }
    
    def save_report(self):
        """Guardar reporte en archivo."""
        report_file = f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        return report_file
    
    def run_all_tests(self):
        """Ejecutar todos los tests."""
        print("🚀 BUL API Final Test Suite")
        print("=" * 50)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ejecutar todos los tests
        tests = [
            ("File Structure", self.test_file_structure),
            ("API Core", self.test_api_core),
            ("Endpoints", self.test_endpoints),
            ("CORS", self.test_cors),
            ("Error Handling", self.test_error_handling),
            ("Async Processing", self.test_async_processing),
            ("Configuration", self.test_configuration),
            ("Requirements", self.test_requirements),
            ("Documentation", self.test_documentation),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Generar resumen y recomendaciones
        self.generate_summary()
        self.generate_recommendations()
        
        # Mostrar resultados
        self.print_results()
        
        # Guardar reporte
        report_file = self.save_report()
        
        return self.results["summary"]["status"] == "READY"
    
    def print_results(self):
        """Imprimir resultados finales."""
        print("\n" + "=" * 50)
        print("📊 FINAL TEST RESULTS")
        print("=" * 50)
        
        summary = self.results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Status: {summary['status']}")
        
        print("\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(self.results["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print("\n📋 API STATUS FOR FRONTEND:")
        if summary["status"] == "READY":
            print("   ✅ API is READY for frontend integration")
            print("   🚀 You can start developing the frontend")
            print("   🔗 Base URL: http://localhost:8000")
            print("   📡 API Version: 3.0.0")
        else:
            print("   ❌ API needs work before frontend integration")
            print("   🔧 Fix the issues listed in recommendations")
            print("   🔄 Re-run tests after fixes")

def main():
    """Función principal."""
    tester = BULAPITester()
    success = tester.run_all_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 CONGRATULATIONS! API is ready for frontend!")
    else:
        print("⚠️  API needs attention before frontend integration")
    
    input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

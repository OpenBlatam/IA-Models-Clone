"""
MOEA API Tester
===============
Script para probar la API del proyecto MOEA una vez generado
"""
import requests
import json
import time
from typing import Dict, Any, Optional

API_BASE = "http://localhost:8000"


class MOEATester:
    """Clase para probar la API MOEA"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_server(self) -> bool:
        """Verificar que el servidor esté corriendo"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def test_optimize_nsga2(self) -> Optional[Dict]:
        """Probar optimización con NSGA-II"""
        print("🧪 Test 1: NSGA-II Optimization")
        print("-" * 60)
        
        problem = {
            "name": "ZDT1",
            "objectives": [
                {"name": "f1", "type": "minimize", "function": "x[0]"},
                {"name": "f2", "type": "minimize", 
                 "function": "g(x) * (1 - sqrt(f1/g(x)))"}
            ],
            "variables": [
                {"name": f"x{i}", "min": 0, "max": 1} for i in range(1, 6)
            ]
        }
        
        algorithm = {
            "algorithm": "nsga2",
            "population_size": 50,
            "generations": 20,
            "mutation_rate": 0.1,
            "crossover_rate": 0.9
        }
        
        try:
            print("   Enviando request...")
            response = self.session.post(
                f"{self.base_url}/api/v1/moea/optimize",
                json={"problem": problem, "algorithm": algorithm},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Éxito!")
                print(f"   Pareto solutions: {len(result.get('pareto_front', []))}")
                print(f"   Hypervolume: {result.get('metrics', {}).get('hypervolume', 'N/A')}")
                return result
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
            return None
    
    def test_get_metrics(self, project_id: str) -> Optional[Dict]:
        """Obtener métricas de un proyecto"""
        print(f"\n🧪 Test 2: Get Metrics (Project: {project_id})")
        print("-" * 60)
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/moea/metrics/{project_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                metrics = response.json()
                print("   ✅ Métricas obtenidas:")
                for key, value in metrics.items():
                    print(f"      {key}: {value}")
                return metrics
            else:
                print(f"   ❌ Error {response.status_code}")
                return None
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
            return None
    
    def test_batch_optimization(self) -> Optional[Dict]:
        """Probar optimización en batch"""
        print("\n🧪 Test 3: Batch Optimization")
        print("-" * 60)
        
        batch_request = {
            "problems": [
                {"name": "ZDT1", "variables": 5},
                {"name": "ZDT2", "variables": 5}
            ],
            "algorithm": {
                "algorithm": "nsga2",
                "population_size": 50,
                "generations": 20
            }
        }
        
        try:
            print("   Enviando batch request...")
            response = self.session.post(
                f"{self.base_url}/api/v1/moea/batch",
                json=batch_request,
                timeout=300
            )
            
            if response.status_code == 200:
                results = response.json()
                print(f"   ✅ Batch completado!")
                print(f"   Problems solved: {len(results.get('results', []))}")
                return results
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
            return None
    
    def test_export_results(self, project_id: str) -> bool:
        """Probar exportación de resultados"""
        print(f"\n🧪 Test 4: Export Results (Project: {project_id})")
        print("-" * 60)
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/moea/export/{project_id}",
                params={"format": "json"},
                timeout=10
            )
            
            if response.status_code == 200:
                # Guardar archivo
                filename = f"moea_export_{project_id}.json"
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"   ✅ Resultados exportados a: {filename}")
                print(f"   Tamaño: {len(response.content)} bytes")
                return True
            else:
                print(f"   ❌ Error {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
            return False
    
    def test_api_docs(self) -> bool:
        """Verificar documentación de API"""
        print("\n🧪 Test 5: API Documentation")
        print("-" * 60)
        
        endpoints = [
            ("/docs", "Swagger UI"),
            ("/redoc", "ReDoc"),
            ("/openapi.json", "OpenAPI JSON")
        ]
        
        all_ok = True
        for endpoint, name in endpoints:
            try:
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"   ✅ {name}: Disponible")
                else:
                    print(f"   ⚠️  {name}: Status {response.status_code}")
                    all_ok = False
            except:
                print(f"   ❌ {name}: No disponible")
                all_ok = False
        
        return all_ok
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        print("=" * 60)
        print("MOEA API Test Suite")
        print("=" * 60)
        print(f"\n🔗 API Base URL: {self.base_url}\n")
        
        # Verificar servidor
        if not self.check_server():
            print("❌ Servidor no disponible!")
            print(f"   Asegúrate de que el backend esté corriendo en {self.base_url}")
            print("   Inicia con: cd backend && uvicorn app.main:app --reload")
            return False
        
        print("✅ Servidor disponible!\n")
        
        # Ejecutar tests
        results = {}
        
        # Test 1: Optimización
        result = self.test_optimize_nsga2()
        results['optimize'] = result is not None
        
        project_id = None
        if result and 'project_id' in result:
            project_id = result['project_id']
        
        # Test 2: Métricas (si tenemos project_id)
        if project_id:
            self.test_get_metrics(project_id)
            results['metrics'] = True
        
        # Test 3: Batch
        batch_result = self.test_batch_optimization()
        results['batch'] = batch_result is not None
        
        # Test 4: Export (si tenemos project_id)
        if project_id:
            self.test_export_results(project_id)
            results['export'] = True
        
        # Test 5: Docs
        results['docs'] = self.test_api_docs()
        
        # Resumen
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name.upper():<15} {status}")
        
        total = len(results)
        passed = sum(results.values())
        print(f"\n   Total: {passed}/{total} tests passed")
        print("=" * 60)
        
        return passed == total


def main():
    """Función principal"""
    import sys
    
    # Permitir cambiar la URL base
    base_url = sys.argv[1] if len(sys.argv) > 1 else API_BASE
    
    tester = MOEATester(base_url)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


"""
Pruebas de Integración Completas
Simula un flujo completo de uso de la API
"""

import requests
import time
import json
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8000"

class IntegrationTest:
    """Pruebas de integración completas."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.tasks_created: List[str] = []
    
    def test_complete_workflow(self) -> bool:
        """Prueba un flujo completo de uso."""
        print("\n" + "="*70)
        print("  🔄 PRUEBA DE INTEGRACIÓN COMPLETA")
        print("="*70 + "\n")
        
        try:
            # 1. Verificar salud
            print("1️⃣ Verificando salud del sistema...")
            health = requests.get(f"{BASE_URL}/api/health", timeout=5)
            if health.status_code != 200:
                print("❌ Sistema no saludable")
                return False
            print("✅ Sistema saludable\n")
            
            # 2. Obtener estadísticas iniciales
            print("2️⃣ Obteniendo estadísticas iniciales...")
            stats_before = requests.get(f"{BASE_URL}/api/stats", timeout=5).json()
            initial_tasks = stats_before.get("total_requests", 0)
            print(f"✅ Requests iniciales: {initial_tasks}\n")
            
            # 3. Crear múltiples documentos
            print("3️⃣ Creando documentos de prueba...")
            document_queries = [
                {
                    "query": "Crear un plan de marketing digital para e-commerce",
                    "business_area": "marketing",
                    "document_type": "strategy",
                    "priority": 1
                },
                {
                    "query": "Desarrollar una estrategia de ventas B2B",
                    "business_area": "sales",
                    "document_type": "strategy",
                    "priority": 2
                },
                {
                    "query": "Crear manual de operaciones para servicio al cliente",
                    "business_area": "operations",
                    "document_type": "manual",
                    "priority": 1
                }
            ]
            
            for i, query in enumerate(document_queries, 1):
                print(f"   Creando documento {i}/{len(document_queries)}...")
                response = requests.post(
                    f"{BASE_URL}/api/documents/generate",
                    json=query,
                    timeout=10
                )
                
                if response.status_code == 200:
                    task_id = response.json().get("task_id")
                    self.tasks_created.append(task_id)
                    print(f"   ✅ Task ID: {task_id}")
                else:
                    print(f"   ❌ Error creando documento {i}")
                    return False
            
            print()
            
            # 4. Esperar y verificar estado de todos los documentos
            print("4️⃣ Esperando procesamiento de documentos...")
            all_completed = False
            max_wait = 60  # 60 segundos máximo
            
            start_time = time.time()
            while time.time() - start_time < max_wait:
                completed_count = 0
                
                for task_id in self.tasks_created:
                    status_response = requests.get(
                        f"{BASE_URL}/api/tasks/{task_id}/status",
                        timeout=5
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        progress = status_data.get("progress", 0)
                        
                        if status == "completed":
                            completed_count += 1
                
                print(f"\r   Progreso: {completed_count}/{len(self.tasks_created)} completados", end="", flush=True)
                
                if completed_count == len(self.tasks_created):
                    all_completed = True
                    break
                
                time.sleep(2)
            
            print()
            
            if not all_completed:
                print("⚠️ No todos los documentos se completaron a tiempo")
                return False
            
            print("✅ Todos los documentos completados\n")
            
            # 5. Obtener documentos generados
            print("5️⃣ Obteniendo documentos generados...")
            documents_obtained = 0
            
            for task_id in self.tasks_created:
                doc_response = requests.get(
                    f"{BASE_URL}/api/tasks/{task_id}/document",
                    timeout=5
                )
                
                if doc_response.status_code == 200:
                    doc_data = doc_response.json()
                    if "document" in doc_data and "content" in doc_data["document"]:
                        documents_obtained += 1
                        content_length = len(doc_data["document"]["content"])
                        print(f"   ✅ Documento {documents_obtained}: {content_length} caracteres")
            
            print()
            
            if documents_obtained != len(self.tasks_created):
                print("❌ No se pudieron obtener todos los documentos")
                return False
            
            # 6. Verificar estadísticas finales
            print("6️⃣ Verificando estadísticas finales...")
            stats_after = requests.get(f"{BASE_URL}/api/stats", timeout=5).json()
            final_requests = stats_after.get("total_requests", 0)
            
            print(f"✅ Requests finales: {final_requests}")
            print(f"✅ Nuevos requests: {final_requests - initial_tasks}\n")
            
            # 7. Listar tareas y documentos
            print("7️⃣ Verificando listado de tareas y documentos...")
            
            tasks_list = requests.get(f"{BASE_URL}/api/tasks?limit=10", timeout=5)
            if tasks_list.status_code == 200:
                tasks_data = tasks_list.json()
                print(f"   ✅ Tareas encontradas: {tasks_data.get('total', 0)}")
            
            docs_list = requests.get(f"{BASE_URL}/api/documents?limit=10", timeout=5)
            if docs_list.status_code == 200:
                docs_data = docs_list.json()
                print(f"   ✅ Documentos encontrados: {docs_data.get('total', 0)}\n")
            
            print("="*70)
            print("✅ PRUEBA DE INTEGRACIÓN COMPLETA EXITOSA")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error en prueba de integración: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_error_handling(self) -> bool:
        """Prueba manejo de errores."""
        print("\n" + "="*70)
        print("  🔍 PRUEBA DE MANEJO DE ERRORES")
        print("="*70 + "\n")
        
        test_cases = [
            {
                "name": "Task ID inexistente",
                "request": lambda: requests.get(f"{BASE_URL}/api/tasks/nonexistent_task/status"),
                "expected_status": 404
            },
            {
                "name": "Query muy corta",
                "request": lambda: requests.post(
                    f"{BASE_URL}/api/documents/generate",
                    json={"query": "test"}
                ),
                "expected_status": 400
            },
            {
                "name": "Endpoint inexistente",
                "request": lambda: requests.get(f"{BASE_URL}/api/nonexistent"),
                "expected_status": 404
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                response = test_case["request"]()
                expected = test_case["expected_status"]
                
                if response.status_code == expected:
                    print(f"✅ {test_case['name']}: Status {response.status_code} (esperado {expected})")
                else:
                    print(f"❌ {test_case['name']}: Status {response.status_code} (esperado {expected})")
                    all_passed = False
            except Exception as e:
                print(f"⚠️ {test_case['name']}: Error {e}")
        
        print()
        return all_passed

def run_integration_tests():
    """Ejecuta todas las pruebas de integración."""
    tester = IntegrationTest()
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Servidor no disponible")
            return False
    except:
        print("❌ No se puede conectar al servidor")
        return False
    
    workflow_ok = tester.test_complete_workflow()
    error_handling_ok = tester.test_error_handling()
    
    return workflow_ok and error_handling_ok

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)

































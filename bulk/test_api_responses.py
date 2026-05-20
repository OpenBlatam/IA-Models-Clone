"""
Script de prueba mejorado para verificar las respuestas de la API
Incluye validaciones, manejo de errores mejorado y reportes detallados
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

# Colorama opcional para colores en terminal
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Crear clases dummy si colorama no está disponible
    class Fore:
        CYAN = GREEN = YELLOW = RED = BLUE = RESET = ""
    class Style:
        RESET_ALL = BRIGHT = ""

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

class Colors:
    """Colores para output."""
    HEADER = Fore.CYAN
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    INFO = Fore.BLUE
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT

class TestResults:
    """Almacena resultados de las pruebas."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.errors: List[str] = []
        self.start_time = datetime.now()
    
    def add_pass(self):
        self.passed += 1
        self.total += 1
    
    def add_fail(self, error: str):
        self.failed += 1
        self.total += 1
        self.errors.append(error)
    
    def print_summary(self):
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"  RESUMEN DE PRUEBAS")
        print(f"{'='*60}{Colors.RESET}")
        print(f"{Colors.INFO}Total de pruebas: {self.total}")
        print(f"{Colors.SUCCESS}Exitosas: {self.passed}")
        print(f"{Colors.ERROR}Fallidas: {self.failed}")
        print(f"{Colors.INFO}Tiempo total: {duration:.2f}s{Colors.RESET}")
        
        if self.errors:
            print(f"\n{Colors.ERROR}Errores encontrados:{Colors.RESET}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        print(f"\n{Colors.SUCCESS}Tasa de éxito: {success_rate:.1f}%{Colors.RESET}\n")

def print_response(title: str, response: requests.Response, results: Optional[TestResults] = None):
    """Imprime una respuesta formateada con validación."""
    status_ok = 200 <= response.status_code < 300
    status_color = Colors.SUCCESS if status_ok else Colors.ERROR
    
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.RESET}")
    print(f"Status Code: {status_color}{response.status_code}{Colors.RESET}")
    
    try:
        data = response.json()
        print(f"{Colors.INFO}Response:{Colors.RESET}")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Validaciones básicas
        if results:
            if status_ok:
                results.add_pass()
                print(f"{Colors.SUCCESS}✓ Test pasado{Colors.RESET}")
            else:
                results.add_fail(f"{title}: Status {response.status_code}")
                print(f"{Colors.ERROR}✗ Test fallido{Colors.RESET}")
    except json.JSONDecodeError:
        print(f"{Colors.WARNING}Response (no JSON):{Colors.RESET}")
        print(response.text[:500])
        if results:
            results.add_fail(f"{title}: Respuesta no es JSON válido")
    except Exception as e:
        print(f"{Colors.ERROR}Error procesando respuesta: {e}{Colors.RESET}")
        if results:
            results.add_fail(f"{title}: {str(e)}")
    
    print()

def check_server():
    """Verifica que el servidor esté disponible."""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}✓ Servidor disponible{Colors.RESET}")
            return True
        else:
            print(f"{Colors.WARNING}⚠ Servidor responde pero con status {response.status_code}{Colors.RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Colors.ERROR}✗ No se puede conectar al servidor{Colors.RESET}")
        print(f"{Colors.INFO}Por favor, inicia el servidor primero:{Colors.RESET}")
        print(f"  python api_frontend_ready.py")
        return False
    except Exception as e:
        print(f"{Colors.ERROR}✗ Error verificando servidor: {e}{Colors.RESET}")
        return False

def test_api():
    """Prueba todos los endpoints de la API."""
    
    results = TestResults()
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"  🧪 Probando API BUL - Frontend Ready")
    print(f"{'='*60}{Colors.RESET}\n")
    
    # Verificar servidor
    if not check_server():
        return results
    
    print(f"{Colors.INFO}Iniciando pruebas...{Colors.RESET}\n")
    
    # 1. Root endpoint
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        print_response("1. Root Endpoint - Información del Sistema", response, results)
        
        # Validar estructura de respuesta
        if response.status_code == 200:
            data = response.json()
            required_fields = ["message", "version", "status"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                results.add_fail(f"Root endpoint: Faltan campos {missing}")
    except Exception as e:
        error_msg = f"Root endpoint: {str(e)}"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    
    # 2. Health check
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        print_response("2. Health Check", response, results)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") != "healthy":
                results.add_fail("Health check: Status no es 'healthy'")
    except Exception as e:
        error_msg = f"Health check: {str(e)}"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    
    # 3. Stats
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        print_response("3. Estadísticas del Sistema", response, results)
        
        if response.status_code == 200:
            data = response.json()
            # Validar que tenga métricas básicas
            expected_fields = ["total_requests", "active_tasks", "success_rate"]
            missing = [f for f in expected_fields if f not in data]
            if missing:
                results.add_fail(f"Stats: Faltan campos {missing}")
    except Exception as e:
        error_msg = f"Stats: {str(e)}"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    
    # 4. Generar documento
    task_id = None
    try:
        document_request = {
            "query": "Crear un plan de marketing digital para una startup tecnológica que vende software SaaS",
            "business_area": "marketing",
            "document_type": "strategy",
            "priority": 1,
            "metadata": {
                "industry": "technology",
                "target_audience": "B2B"
            }
        }
        
        print(f"{Colors.INFO}Enviando solicitud de generación de documento...{Colors.RESET}")
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json=document_request,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        print_response("4. Generar Documento - Request", response, results)
        
        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            
            # Validar respuesta
            required_fields = ["task_id", "status", "message", "created_at"]
            missing = [f for f in required_fields if f not in task_data]
            if missing:
                results.add_fail(f"Generate document: Faltan campos {missing}")
            
            if task_id:
                print(f"{Colors.SUCCESS}✓ Task ID generado: {task_id}{Colors.RESET}\n")
                
                # 5. Verificar estado de la tarea
                print(f"{Colors.INFO}⏳ Esperando procesamiento...{Colors.RESET}")
                max_attempts = 30  # 30 segundos máximo
                completed = False
                
                for attempt in range(max_attempts):
                    try:
                        status_response = requests.get(
                            f"{BASE_URL}/api/tasks/{task_id}/status",
                            timeout=TIMEOUT
                        )
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            status = status_data.get("status")
                            progress = status_data.get("progress", 0)
                            
                            # Barra de progreso
                            bar_length = 40
                            filled = int(bar_length * progress / 100)
                            bar = "█" * filled + "░" * (bar_length - filled)
                            print(f"\r{Colors.INFO}   [{bar}] {progress}% - {status}{Colors.RESET}", end="", flush=True)
                            
                            if status == "completed":
                                print(f"\n{Colors.SUCCESS}✓ Documento completado{Colors.RESET}")
                                print_response("5. Estado de Tarea - Completada", status_response, results)
                                
                                # Validar resultado
                                if "result" in status_data and status_data["result"]:
                                    result = status_data["result"]
                                    if "content" in result:
                                        word_count = result.get("word_count", 0)
                                        print(f"{Colors.INFO}   Palabras generadas: {word_count}{Colors.RESET}")
                                
                                # 6. Obtener documento generado
                                try:
                                    doc_response = requests.get(
                                        f"{BASE_URL}/api/tasks/{task_id}/document",
                                        timeout=TIMEOUT
                                    )
                                    print_response("6. Documento Generado", doc_response, results)
                                    
                                    if doc_response.status_code == 200:
                                        doc_data = doc_response.json()
                                        if "document" in doc_data and "content" in doc_data["document"]:
                                            content = doc_data["document"]["content"]
                                            print(f"{Colors.SUCCESS}✓ Documento recibido: {len(content)} caracteres{Colors.RESET}")
                                            print(f"{Colors.INFO}Vista previa (primeros 200 caracteres):{Colors.RESET}")
                                            print(f"{Colors.BOLD}{content[:200]}...{Colors.RESET}\n")
                                except Exception as e:
                                    error_msg = f"Obtener documento: {str(e)}"
                                    print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
                                    results.add_fail(error_msg)
                                
                                completed = True
                                break
                            elif status == "failed":
                                print(f"\n{Colors.ERROR}✗ Documento falló{Colors.RESET}")
                                print_response("5. Estado de Tarea - Fallida", status_response, results)
                                results.add_fail(f"Documento falló: {status_data.get('error', 'Unknown error')}")
                                completed = True
                                break
                        
                        time.sleep(1)
                    except Exception as e:
                        print(f"\n{Colors.WARNING}⚠ Error consultando estado: {e}{Colors.RESET}")
                        time.sleep(1)
                
                if not completed:
                    results.add_fail("Timeout esperando completación del documento")
                    print(f"\n{Colors.WARNING}⚠ Timeout esperando completación{Colors.RESET}")
            else:
                results.add_fail("Generate document: No se recibió task_id")
        else:
            results.add_fail(f"Generate document: Status {response.status_code}")
            
    except requests.exceptions.Timeout:
        error_msg = "Generar documento: Timeout"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    except Exception as e:
        error_msg = f"Generar documento: {str(e)}"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    
    # 7. Listar tareas
    try:
        response = requests.get(f"{BASE_URL}/api/tasks?limit=5", timeout=TIMEOUT)
        print_response("7. Listar Tareas", response, results)
        
        if response.status_code == 200:
            data = response.json()
            if "tasks" in data and isinstance(data["tasks"], list):
                print(f"{Colors.INFO}   Total de tareas: {data.get('total', 0)}{Colors.RESET}")
    except Exception as e:
        error_msg = f"Listar tareas: {str(e)}"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    
    # 8. Listar documentos
    try:
        response = requests.get(f"{BASE_URL}/api/documents?limit=5", timeout=TIMEOUT)
        print_response("8. Listar Documentos", response, results)
        
        if response.status_code == 200:
            data = response.json()
            if "documents" in data and isinstance(data["documents"], list):
                print(f"{Colors.INFO}   Total de documentos: {data.get('total', 0)}{Colors.RESET}")
    except Exception as e:
        error_msg = f"Listar documentos: {str(e)}"
        print(f"{Colors.ERROR}✗ {error_msg}{Colors.RESET}\n")
        results.add_fail(error_msg)
    
    # 9. Probar validaciones (campos faltantes)
    try:
        print(f"{Colors.INFO}Probando validaciones...{Colors.RESET}")
        invalid_request = {"query": "test"}  # Query muy corta o campos faltantes
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json=invalid_request,
            timeout=TIMEOUT
        )
        if response.status_code == 400:
            print(f"{Colors.SUCCESS}✓ Validación funcionando correctamente{Colors.RESET}\n")
            results.add_pass()
        elif response.status_code == 422:
            # FastAPI validation error (422)
            print(f"{Colors.SUCCESS}✓ Validación funcionando correctamente (422){Colors.RESET}\n")
            results.add_pass()
        else:
            # Si acepta la request inválida, es un problema
            print(f"{Colors.WARNING}⚠ Validación: Status {response.status_code} (esperado 400 o 422){Colors.RESET}\n")
            results.add_fail(f"Validación: Debería rechazar query muy corta (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"{Colors.WARNING}⚠ No se pudo probar validación: Servidor no disponible{Colors.RESET}\n")
        results.add_fail("Validación: No se pudo conectar al servidor")
    except Exception as e:
        print(f"{Colors.WARNING}⚠ No se pudo probar validación: {e}{Colors.RESET}\n")
        results.add_fail(f"Validación: Error - {str(e)}")
    
    return results

if __name__ == "__main__":
    print(f"\n{Colors.HEADER}{Colors.BOLD}╔════════════════════════════════════════════════════════════╗")
    print(f"║  Script de Pruebas - API BUL Frontend Ready          ║")
    print(f"╚════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"\n{Colors.INFO}⚠️  Asegúrate de que el servidor esté corriendo en {BASE_URL}{Colors.RESET}")
    print(f"{Colors.INFO}   Ejecuta: python api_frontend_ready.py{Colors.RESET}\n")
    
    try:
        results = test_api()
        results.print_summary()
        
        # Exit code basado en resultados
        if results.failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠ Pruebas interrumpidas por el usuario{Colors.RESET}\n")
        sys.exit(130)
    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.ERROR}✗ Error: No se puede conectar al servidor.{Colors.RESET}")
        print(f"{Colors.INFO}Por favor, inicia el servidor primero:{Colors.RESET}")
        print(f"  python api_frontend_ready.py\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.ERROR}✗ Error inesperado: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


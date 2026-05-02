"""
Script de pruebas avanzadas para la API BUL
Incluye: pruebas de carga, WebSocket, exportación de resultados, y más
"""

import requests
import json
import time
import sys
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import csv

# Websockets opcional
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

# Colorama opcional
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class Fore:
        CYAN = GREEN = YELLOW = RED = BLUE = MAGENTA = RESET = ""
    class Style:
        RESET_ALL = BRIGHT = ""

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
TIMEOUT = 30

class Colors:
    HEADER = Fore.CYAN
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    INFO = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT

class AdvancedTestResults:
    """Resultados avanzados con exportación."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.errors: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.tests: List[Dict[str, Any]] = []
    
    def add_test(self, name: str, passed: bool, duration: float, details: Dict[str, Any] = None):
        """Agrega un resultado de prueba."""
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        test_result = {
            "name": name,
            "passed": passed,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.tests.append(test_result)
        
        if not passed:
            self.errors.append({
                "test": name,
                "error": details.get("error", "Unknown error") if details else "Unknown error"
            })
    
    def export_json(self, filename: str = "test_results.json"):
        """Exporta resultados a JSON."""
        results = {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": (self.passed / self.total * 100) if self.total > 0 else 0,
                "duration": (datetime.now() - self.start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            },
            "tests": self.tests,
            "errors": self.errors,
            "metrics": self.metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"{Colors.SUCCESS}✓ Resultados exportados a {filename}{Colors.RESET}")
        return filename
    
    def export_csv(self, filename: str = "test_results.csv"):
        """Exporta resultados a CSV."""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Test', 'Status', 'Duration (s)', 'Timestamp', 'Details'])
            
            for test in self.tests:
                writer.writerow([
                    test['name'],
                    'PASS' if test['passed'] else 'FAIL',
                    test['duration'],
                    test['timestamp'],
                    json.dumps(test.get('details', {}))
                ])
        
        print(f"{Colors.SUCCESS}✓ Resultados exportados a {filename}{Colors.RESET}")
        return filename
    
    def print_summary(self):
        """Imprime resumen mejorado."""
        duration = (datetime.now() - self.start_time).total_seconds()
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
        print(f"  RESUMEN DE PRUEBAS AVANZADAS")
        print(f"{'='*70}{Colors.RESET}")
        print(f"{Colors.INFO}Total de pruebas: {self.total}")
        print(f"{Colors.SUCCESS}Exitosas: {self.passed}")
        print(f"{Colors.ERROR}Fallidas: {self.failed}")
        print(f"{Colors.INFO}Tiempo total: {duration:.2f}s")
        print(f"{Colors.MAGENTA}Tasa de éxito: {success_rate:.1f}%{Colors.RESET}")
        
        if self.metrics:
            print(f"\n{Colors.INFO}Métricas:{Colors.RESET}")
            for key, value in self.metrics.items():
                print(f"  {key}: {value}")
        
        if self.errors:
            print(f"\n{Colors.ERROR}Errores encontrados:{Colors.RESET}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error['test']}: {error['error']}")
        
        print()

def test_websocket(task_id: str, results: AdvancedTestResults) -> bool:
    """Prueba conexión WebSocket."""
    start_time = time.time()
    
    if not WEBSOCKETS_AVAILABLE:
        duration = time.time() - start_time
        results.add_test("WebSocket Connection", False, duration, {"error": "websockets module not available"})
        print(f"{Colors.WARNING}⚠ WebSocket test skipped: websockets module not installed{Colors.RESET}")
        print(f"{Colors.INFO}   Install with: pip install websockets{Colors.RESET}")
        return False
    
    try:
        async def test_ws():
            uri = f"{WS_URL}/api/ws/{task_id}"
            try:
                async with websockets.connect(uri, timeout=10) as ws:
                    # Esperar mensaje inicial
                    message = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(message)
                    
                    if data.get("type") in ["initial_state", "connected", "status"]:
                        return True
                    return False
            except asyncio.TimeoutError:
                print(f"{Colors.WARNING}⚠ WebSocket timeout{Colors.RESET}")
                return False
            except Exception as e:
                print(f"{Colors.ERROR}Error WebSocket: {e}{Colors.RESET}")
                return False
        
        result = asyncio.run(test_ws())
        duration = time.time() - start_time
        
        if result:
            results.add_test("WebSocket Connection", True, duration)
            print(f"{Colors.SUCCESS}✓ WebSocket conectado exitosamente{Colors.RESET}")
            return True
        else:
            results.add_test("WebSocket Connection", False, duration, {"error": "No se recibió mensaje válido o timeout"})
            print(f"{Colors.WARNING}⚠ WebSocket no recibió mensaje válido{Colors.RESET}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        results.add_test("WebSocket Connection", False, duration, {"error": str(e)})
        print(f"{Colors.ERROR}✗ Error en WebSocket: {e}{Colors.RESET}")
        return False

def test_load(duration_seconds: int = 10, requests_per_second: int = 2, results: AdvancedTestResults = None):
    """Prueba de carga básica."""
    print(f"\n{Colors.INFO}Iniciando prueba de carga...{Colors.RESET}")
    print(f"{Colors.INFO}Duración: {duration_seconds}s, RPS: {requests_per_second}{Colors.RESET}\n")
    
    start_time = time.time()
    requests_sent = 0
    requests_success = 0
    requests_failed = 0
    response_times = []
    
    end_time = start_time + duration_seconds
    interval = 1.0 / requests_per_second
    
    try:
        while time.time() < end_time:
            request_start = time.time()
            try:
                response = requests.get(f"{BASE_URL}/api/health", timeout=5)
                request_duration = time.time() - request_start
                response_times.append(request_duration)
                requests_sent += 1
                
                if response.status_code == 200:
                    requests_success += 1
                else:
                    requests_failed += 1
            except Exception as e:
                requests_failed += 1
                requests_sent += 1
            
            time.sleep(max(0, interval - (time.time() - request_start)))
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Prueba de carga interrumpida{Colors.RESET}")
    
    total_duration = time.time() - start_time
    
    # Calcular métricas
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
    else:
        avg_response_time = min_response_time = max_response_time = 0
    
    success_rate = (requests_success / requests_sent * 100) if requests_sent > 0 else 0
    actual_rps = requests_sent / total_duration if total_duration > 0 else 0
    
    metrics = {
        "total_requests": requests_sent,
        "successful": requests_success,
        "failed": requests_failed,
        "success_rate": f"{success_rate:.1f}%",
        "avg_response_time": f"{avg_response_time*1000:.2f}ms",
        "min_response_time": f"{min_response_time*1000:.2f}ms",
        "max_response_time": f"{max_response_time*1000:.2f}ms",
        "actual_rps": f"{actual_rps:.2f}",
        "duration": f"{total_duration:.2f}s"
    }
    
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f"  RESULTADOS DE PRUEBA DE CARGA")
    print(f"{'='*70}{Colors.RESET}")
    print(f"{Colors.INFO}Total de requests: {requests_sent}")
    print(f"{Colors.SUCCESS}Exitosos: {requests_success}")
    print(f"{Colors.ERROR}Fallidos: {requests_failed}")
    print(f"{Colors.MAGENTA}Tasa de éxito: {success_rate:.1f}%{Colors.RESET}")
    print(f"{Colors.INFO}Tiempo promedio: {avg_response_time*1000:.2f}ms")
    print(f"{Colors.INFO}Tiempo mínimo: {min_response_time*1000:.2f}ms")
    print(f"{Colors.INFO}Tiempo máximo: {max_response_time*1000:.2f}ms")
    print(f"{Colors.INFO}RPS real: {actual_rps:.2f}{Colors.RESET}\n")
    
    if results:
        results.metrics["load_test"] = metrics
        results.add_test("Load Test", requests_failed == 0, total_duration, metrics)
    
    return metrics

def test_concurrent_requests(count: int = 5, results: AdvancedTestResults = None):
    """Prueba requests concurrentes."""
    print(f"\n{Colors.INFO}Probando {count} requests concurrentes...{Colors.RESET}")
    
    start_time = time.time()
    
    def make_request():
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    # Simular requests concurrentes (secuencial por ahora, se puede mejorar con threading)
    results_list = []
    for i in range(count):
        results_list.append(make_request())
        time.sleep(0.1)  # Pequeño delay
    
    duration = time.time() - start_time
    success_count = sum(results_list)
    success_rate = (success_count / count * 100) if count > 0 else 0
    
    print(f"{Colors.INFO}Requests concurrentes: {count}")
    print(f"{Colors.SUCCESS}Exitosos: {success_count}")
    print(f"{Colors.ERROR}Fallidos: {count - success_count}")
    print(f"{Colors.MAGENTA}Tasa de éxito: {success_rate:.1f}%{Colors.RESET}")
    print(f"{Colors.INFO}Tiempo total: {duration:.2f}s{Colors.RESET}\n")
    
    if results:
        results.add_test("Concurrent Requests", success_count == count, duration, {
            "total": count,
            "successful": success_count,
            "failed": count - success_count,
            "success_rate": f"{success_rate:.1f}%"
        })
    
    return success_count == count

def test_rate_limiting(results: AdvancedTestResults = None):
    """Prueba rate limiting."""
    print(f"\n{Colors.INFO}Probando rate limiting (10 req/min)...{Colors.RESET}")
    
    start_time = time.time()
    requests_sent = 0
    rate_limited = False
    
    # Enviar más de 10 requests rápidamente
    for i in range(15):
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": f"Test query {i} para rate limiting"},
                timeout=5
            )
            requests_sent += 1
            
            if response.status_code == 429:  # Too Many Requests
                rate_limited = True
                print(f"{Colors.SUCCESS}✓ Rate limiting funcionando (request {i+1} limitado){Colors.RESET}")
                break
            
            time.sleep(0.1)
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Error en request {i+1}: {e}{Colors.RESET}")
    
    duration = time.time() - start_time
    
    if results:
        results.add_test("Rate Limiting", rate_limited, duration, {
            "requests_sent": requests_sent,
            "rate_limited": rate_limited
        })
    
    if not rate_limited:
        print(f"{Colors.WARNING}⚠ Rate limiting no se activó después de {requests_sent} requests{Colors.RESET}")
    
    return rate_limited

def run_advanced_tests():
    """Ejecuta todas las pruebas avanzadas."""
    results = AdvancedTestResults()
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  🧪 PRUEBAS AVANZADAS - API BUL")
    print(f"{'='*70}{Colors.RESET}\n")
    
    # Verificar servidor
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"{Colors.ERROR}✗ Servidor no disponible{Colors.RESET}")
            return results
        print(f"{Colors.SUCCESS}✓ Servidor disponible{Colors.RESET}\n")
    except Exception as e:
        print(f"{Colors.ERROR}✗ No se puede conectar al servidor: {e}{Colors.RESET}")
        return results
    
    # 1. Prueba de carga
    test_load(duration_seconds=5, requests_per_second=2, results=results)
    
    # 2. Prueba de requests concurrentes
    test_concurrent_requests(count=5, results=results)
    
    # 3. Prueba de rate limiting
    test_rate_limiting(results=results)
    
    # 4. Generar documento y probar WebSocket
    if WEBSOCKETS_AVAILABLE:
        try:
            print(f"{Colors.INFO}Generando documento para probar WebSocket...{Colors.RESET}")
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={
                    "query": "Test document para WebSocket",
                    "priority": 1
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                if task_id:
                    # Probar WebSocket
                    test_websocket(task_id, results)
                else:
                    print(f"{Colors.WARNING}⚠ No se recibió task_id para WebSocket{Colors.RESET}")
            else:
                print(f"{Colors.WARNING}⚠ No se pudo generar documento para WebSocket (status: {response.status_code}){Colors.RESET}")
        except requests.exceptions.ConnectionError:
            print(f"{Colors.WARNING}⚠ Servidor no disponible para probar WebSocket{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ No se pudo probar WebSocket: {e}{Colors.RESET}")
    else:
        print(f"{Colors.INFO}⚠ WebSocket test skipped (websockets module not available){Colors.RESET}")
    
    # Exportar resultados
    print(f"\n{Colors.INFO}Exportando resultados...{Colors.RESET}")
    json_file = results.export_json()
    csv_file = results.export_csv()
    
    # Generar dashboard HTML
    try:
        from test_dashboard_generator import generate_html_dashboard
        dashboard_file = generate_html_dashboard(json_file, "test_dashboard.html")
        print(f"{Colors.SUCCESS}✓ Dashboard generado: {dashboard_file}{Colors.RESET}")
    except ImportError:
        print(f"{Colors.WARNING}⚠ No se pudo generar dashboard (módulo no disponible){Colors.RESET}")
    except Exception as e:
        print(f"{Colors.WARNING}⚠ Error generando dashboard: {e}{Colors.RESET}")
    
    # Mostrar resumen
    results.print_summary()
    
    return results

if __name__ == "__main__":
    print(f"\n{Colors.HEADER}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Pruebas Avanzadas - API BUL Frontend Ready          ║")
    print(f"╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"\n{Colors.INFO}⚠️  Asegúrate de que el servidor esté corriendo en {BASE_URL}{Colors.RESET}\n")
    
    try:
        results = run_advanced_tests()
        sys.exit(0 if results.failed == 0 else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠ Pruebas interrumpidas{Colors.RESET}\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.ERROR}✗ Error: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


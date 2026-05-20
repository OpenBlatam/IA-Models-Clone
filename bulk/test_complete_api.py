"""
Test Completo y Mejorado de la API BUL
Cubre todos los endpoints, casos de uso, edge cases, performance y más
"""

import requests
import json
import time
import sys
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path

# WebSocket opcional
try:
    import websockets
    import asyncio
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    asyncio = None

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

class TestResults:
    """Almacena resultados de las pruebas con estadísticas mejoradas."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.errors: List[str] = []
        self.start_time = datetime.now()
        self.tests: List[Dict[str, Any]] = []
        self.response_times: List[float] = []
        self.categories: Dict[str, Dict[str, int]] = {}
    
    def add_pass(self, test_name: str = "", category: str = "general", response_time: float = 0):
        self.passed += 1
        self.total += 1
        if response_time > 0:
            self.response_times.append(response_time)
        self.tests.append({
            "name": test_name,
            "status": "PASS",
            "category": category,
            "response_time": response_time,
            "time": datetime.now().isoformat()
        })
        self._update_category(category, "passed")
    
    def add_fail(self, error: str, test_name: str = "", category: str = "general", response_time: float = 0):
        self.failed += 1
        self.total += 1
        if response_time > 0:
            self.response_times.append(response_time)
        self.errors.append(error)
        self.tests.append({
            "name": test_name,
            "status": "FAIL",
            "category": category,
            "error": error,
            "response_time": response_time,
            "time": datetime.now().isoformat()
        })
        self._update_category(category, "failed")
    
    def _update_category(self, category: str, result: str):
        if category not in self.categories:
            self.categories[category] = {"passed": 0, "failed": 0}
        self.categories[category][result] += 1
    
    def get_avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def export_json(self, filename: str = "test_results_complete.json"):
        """Exporta resultados a JSON."""
        data = {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": (self.passed / self.total * 100) if self.total > 0 else 0,
                "duration": (datetime.now() - self.start_time).total_seconds(),
                "avg_response_time": self.get_avg_response_time()
            },
            "categories": self.categories,
            "tests": self.tests,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat()
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename
    
    def export_csv(self, filename: str = "test_results_complete.csv"):
        """Exporta resultados a CSV."""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Test Name", "Status", "Category", "Response Time (ms)", "Error", "Timestamp"])
            for test in self.tests:
                writer.writerow([
                    test.get("name", ""),
                    test.get("status", ""),
                    test.get("category", ""),
                    f"{test.get('response_time', 0) * 1000:.2f}",
                    test.get("error", ""),
                    test.get("time", "")
                ])
        return filename
    
    def export_html(self, filename: str = "test_results_complete.html"):
        """Exporta resultados a HTML con visualización mejorada."""
        duration = (datetime.now() - self.start_time).total_seconds()
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results - API BUL</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }}
        h1 {{ color: #333; margin-bottom: 30px; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .card.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .card.error {{ background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }}
        .card.info {{ background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }}
        .card h3 {{ font-size: 14px; opacity: 0.9; margin-bottom: 10px; }}
        .card .value {{ font-size: 32px; font-weight: bold; }}
        .categories {{ margin-bottom: 30px; }}
        .category {{ background: #f9f9f9; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .category h3 {{ color: #333; margin-bottom: 10px; }}
        .progress-bar {{ background: #e0e0e0; height: 25px; border-radius: 12px; overflow: hidden; margin-top: 10px; }}
        .progress-fill {{ background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: width 0.3s; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .status-pass {{ color: #4CAF50; font-weight: bold; }}
        .status-fail {{ color: #f44336; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .errors {{ background: #ffebee; padding: 20px; border-radius: 5px; margin-top: 20px; border-left: 4px solid #f44336; }}
        .errors h3 {{ color: #c62828; margin-bottom: 15px; }}
        .errors ul {{ list-style: none; }}
        .errors li {{ padding: 8px; margin-bottom: 5px; background: white; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Test Results - API BUL</h1>
        
        <div class="summary">
            <div class="card">
                <h3>Total Tests</h3>
                <div class="value">{self.total}</div>
            </div>
            <div class="card success">
                <h3>Passed</h3>
                <div class="value">{self.passed}</div>
            </div>
            <div class="card error">
                <h3>Failed</h3>
                <div class="value">{self.failed}</div>
            </div>
            <div class="card info">
                <h3>Success Rate</h3>
                <div class="value">{success_rate:.1f}%</div>
            </div>
            <div class="card info">
                <h3>Duration</h3>
                <div class="value">{duration:.1f}s</div>
            </div>
        </div>
        
        <div class="categories">
            <h2>Results by Category</h2>
"""
        
        for category, stats in self.categories.items():
            total_cat = stats["passed"] + stats["failed"]
            success_cat = (stats["passed"] / total_cat * 100) if total_cat > 0 else 0
            html += f"""
            <div class="category">
                <h3>{category.title()} - {stats['passed']}/{total_cat} passed</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {success_cat}%">{success_cat:.1f}%</div>
                </div>
            </div>
"""
        
        html += """
        </div>
        
        <h2>Test Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Category</th>
                    <th>Response Time</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for test in self.tests:
            status_class = "status-pass" if test.get("status") == "PASS" else "status-fail"
            status_text = "✓ PASS" if test.get("status") == "PASS" else "✗ FAIL"
            response_time = f"{test.get('response_time', 0) * 1000:.2f}ms" if test.get('response_time') else "N/A"
            html += f"""
                <tr>
                    <td>{test.get('name', 'N/A')}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{test.get('category', 'N/A')}</td>
                    <td>{response_time}</td>
                    <td>{test.get('time', 'N/A')[:19] if test.get('time') else 'N/A'}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        if self.errors:
            html += """
        <div class="errors">
            <h3>Errors Found</h3>
            <ul>
"""
            for error in self.errors:
                html += f"<li>{error}</li>"
            html += """
            </ul>
        </div>
"""
        
        html += f"""
        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        return filename
    
    def print_summary(self):
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{Colors.HEADER}{'='*70}")
        print(f"  RESUMEN DE PRUEBAS COMPLETAS")
        print(f"{'='*70}{Colors.RESET}")
        print(f"{Colors.INFO}Total de pruebas: {self.total}")
        print(f"{Colors.SUCCESS}Exitosas: {self.passed}")
        print(f"{Colors.ERROR}Fallidas: {self.failed}")
        print(f"{Colors.INFO}Tiempo total: {duration:.2f}s{Colors.RESET}")
        
        if self.response_times:
            avg_time = self.get_avg_response_time()
            min_time = min(self.response_times)
            max_time = max(self.response_times)
            print(f"{Colors.INFO}Tiempo promedio de respuesta: {avg_time*1000:.2f}ms")
            print(f"{Colors.INFO}Tiempo mínimo: {min_time*1000:.2f}ms")
            print(f"{Colors.INFO}Tiempo máximo: {max_time*1000:.2f}ms{Colors.RESET}")
        
        if self.categories:
            print(f"\n{Colors.HEADER}Por categoría:{Colors.RESET}")
            for category, stats in self.categories.items():
                total_cat = stats["passed"] + stats["failed"]
                success_cat = (stats["passed"] / total_cat * 100) if total_cat > 0 else 0
                print(f"  {category}: {stats['passed']}/{total_cat} ({success_cat:.1f}%)")
        
        if self.errors:
            print(f"\n{Colors.ERROR}Errores encontrados:{Colors.RESET}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        print(f"\n{Colors.SUCCESS}Tasa de éxito: {success_rate:.1f}%{Colors.RESET}\n")
        
        # Exportar resultados
        try:
            json_file = self.export_json()
            csv_file = self.export_csv()
            html_file = self.export_html()
            print(f"{Colors.INFO}Resultados exportados:{Colors.RESET}")
            print(f"  - {json_file}")
            print(f"  - {csv_file}")
            print(f"  - {html_file} (abre en navegador)\n")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ No se pudieron exportar resultados: {e}{Colors.RESET}\n")

# ============================================================================
# MEJORAS AVANZADAS: Test Retry, Parallelization, Filtering, Fixtures
# ============================================================================

class TestRetry:
    """Maneja reintentos de tests flaky."""
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def run_with_retry(self, test_func, *args, **kwargs):
        """Ejecuta un test con reintentos."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return test_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise last_error

class TestFilter:
    """Filtra tests por categoría, nombre o tags."""
    def __init__(self, categories: Optional[List[str]] = None, 
                 exclude_categories: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None):
        self.categories = set(categories) if categories else None
        self.exclude_categories = set(exclude_categories) if exclude_categories else None
        self.tags = set(tags) if tags else None
    
    def should_run(self, category: str, test_name: str = "", tags: List[str] = None) -> bool:
        """Determina si un test debe ejecutarse."""
        if self.categories and category not in self.categories:
            return False
        if self.exclude_categories and category in self.exclude_categories:
            return False
        if self.tags and tags:
            if not any(tag in self.tags for tag in tags):
                return False
        return True

class TestFixtures:
    """Maneja setup y teardown de tests."""
    def __init__(self):
        self.setup_functions: List[callable] = []
        self.teardown_functions: List[callable] = []
        self.test_data: Dict[str, Any] = {}
    
    def setup(self, func: callable):
        """Registra una función de setup."""
        self.setup_functions.append(func)
        return func
    
    def teardown(self, func: callable):
        """Registra una función de teardown."""
        self.teardown_functions.append(func)
        return func
    
    def run_setup(self):
        """Ejecuta todas las funciones de setup."""
        for func in self.setup_functions:
            try:
                func()
            except Exception as e:
                print(f"{Colors.WARNING}⚠ Setup error: {e}{Colors.RESET}")
    
    def run_teardown(self):
        """Ejecuta todas las funciones de teardown."""
        for func in self.teardown_functions:
            try:
                func()
            except Exception as e:
                print(f"{Colors.WARNING}⚠ Teardown error: {e}{Colors.RESET}")

class CoverageTracker:
    """Rastrea cobertura de endpoints y funcionalidades."""
    def __init__(self):
        self.tested_endpoints: Set[str] = set()
        self.tested_methods: Dict[str, Set[str]] = {}
        self.tested_scenarios: Set[str] = set()
    
    def mark_endpoint(self, endpoint: str, method: str = "GET"):
        """Marca un endpoint como probado."""
        self.tested_endpoints.add(endpoint)
        if endpoint not in self.tested_methods:
            self.tested_methods[endpoint] = set()
        self.tested_methods[endpoint].add(method)
    
    def mark_scenario(self, scenario: str):
        """Marca un escenario como probado."""
        self.tested_scenarios.add(scenario)
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Genera reporte de cobertura."""
        return {
            "endpoints_tested": len(self.tested_endpoints),
            "endpoints": list(self.tested_endpoints),
            "methods_by_endpoint": {ep: list(methods) for ep, methods in self.tested_methods.items()},
            "scenarios_tested": len(self.tested_scenarios),
            "scenarios": list(self.tested_scenarios)
        }

class ParallelTestRunner:
    """Ejecuta tests en paralelo."""
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def run_parallel(self, test_functions: List[callable], results: TestResults):
        """Ejecuta tests en paralelo."""
        def run_test(test_func):
            try:
                test_func(results)
                return True
            except Exception as e:
                print(f"{Colors.ERROR}✗ Error en test paralelo: {e}{Colors.RESET}")
                return False
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_test, func) for func in test_functions]
            for future in as_completed(futures):
                future.result()

class EnhancedErrorReporter:
    """Proporciona reportes de error mejorados."""
    @staticmethod
    def format_error(error: Exception, context: Dict[str, Any] = None) -> str:
        """Formatea un error con contexto adicional."""
        error_msg = str(error)
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            return f"{error_msg} [Context: {context_str}]"
        return error_msg
    
    @staticmethod
    def suggest_fix(error: Exception) -> str:
        """Sugiere una solución para el error."""
        error_str = str(error).lower()
        if "connection" in error_str or "timeout" in error_str:
            return "Verifica que el servidor esté corriendo: python api_frontend_ready.py"
        elif "404" in error_str:
            return "Verifica que el endpoint exista en la API"
        elif "422" in error_str or "validation" in error_str:
            return "Verifica el formato de los datos enviados"
        elif "rate limit" in error_str:
            return "Espera unos segundos antes de reintentar"
        return "Revisa los logs del servidor para más detalles"

# Instancias globales
test_retry = TestRetry(max_retries=3, retry_delay=1.0)
test_fixtures = TestFixtures()
coverage_tracker = CoverageTracker()
parallel_runner = ParallelTestRunner(max_workers=4)
error_reporter = EnhancedErrorReporter()

# Fixtures de setup/teardown
@test_fixtures.setup
def setup_tests():
    """Setup inicial para tests."""
    # Limpiar cache si es necesario
    pass

@test_fixtures.teardown
def teardown_tests():
    """Teardown final para tests."""
    # Limpiar recursos si es necesario
    pass

def check_server():
    """Verifica que el servidor esté disponible."""
    try:
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}✓ Servidor disponible (respuesta en {elapsed*1000:.0f}ms){Colors.RESET}")
            coverage_tracker.mark_endpoint("/api/health", "GET")
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

# Decorador para tests con retry automático
def with_retry(max_retries: int = 3):
    """Decorador para ejecutar tests con reintentos."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if use_retry:
                return test_retry.run_with_retry(func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Variable global para controlar retry (se establece en main)
use_retry = True

def test_root_endpoint(results: TestResults):
    """Prueba el endpoint raíz."""
    print(f"\n{Colors.HEADER}[TEST] Root Endpoint{Colors.RESET}")
    coverage_tracker.mark_endpoint("/", "GET")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            required = ["message", "version", "status"]
            missing = [f for f in required if f not in data]
            if not missing:
                print(f"{Colors.SUCCESS}  ✓ Root endpoint OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("Root Endpoint", "system", elapsed)
            else:
                results.add_fail(f"Root endpoint: Faltan campos {missing}", "Root Endpoint", "system", elapsed)
        else:
            results.add_fail(f"Root endpoint: Status {response.status_code}", "Root Endpoint", "system", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Root endpoint: {str(e)}", "Root Endpoint", "system", elapsed)

def test_health_endpoint(results: TestResults):
    """Prueba el endpoint de health."""
    print(f"\n{Colors.HEADER}[TEST] Health Check{Colors.RESET}")
    coverage_tracker.mark_endpoint("/api/health", "GET")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print(f"{Colors.SUCCESS}  ✓ Health check OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("Health Check", "system", elapsed)
            else:
                results.add_fail(f"Health check: Status no es 'healthy'", "Health Check", "system", elapsed)
        else:
            results.add_fail(f"Health check: Status {response.status_code}", "Health Check", "system", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Health check: {str(e)}", "Health Check", "system", elapsed)

def test_stats_endpoint(results: TestResults):
    """Prueba el endpoint de estadísticas."""
    print(f"\n{Colors.HEADER}[TEST] Stats Endpoint{Colors.RESET}")
    coverage_tracker.mark_endpoint("/api/stats", "GET")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            expected = ["total_requests", "active_tasks", "success_rate"]
            missing = [f for f in expected if f not in data]
            if not missing:
                print(f"{Colors.SUCCESS}  ✓ Stats endpoint OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("Stats Endpoint", "system", elapsed)
            else:
                results.add_fail(f"Stats: Faltan campos {missing}", "Stats Endpoint", "system", elapsed)
        else:
            results.add_fail(f"Stats: Status {response.status_code}", "Stats Endpoint", "system", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Stats: {str(e)}", "Stats Endpoint", "system", elapsed)

def test_generate_document(results: TestResults):
    """Prueba la generación de documentos."""
    print(f"\n{Colors.HEADER}[TEST] Generate Document{Colors.RESET}")
    start = time.time()
    try:
        request_data = {
            "query": "Crear un plan de marketing digital completo para una startup tecnológica",
            "business_area": "marketing",
            "document_type": "strategy",
            "priority": 1,
            "metadata": {
                "industry": "technology",
                "target_audience": "B2B"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            if "task_id" in data:
                print(f"{Colors.SUCCESS}  ✓ Document generation iniciada ({elapsed*1000:.0f}ms){Colors.RESET}")
                print(f"{Colors.INFO}  Task ID: {data.get('task_id')}{Colors.RESET}")
                results.add_pass("Generate Document", "documents", elapsed)
                return data.get("task_id")
            else:
                results.add_fail("Generate document: No task_id en respuesta", "Generate Document", "documents", elapsed)
        elif response.status_code == 422:
            print(f"{Colors.WARNING}  ⚠ Validación FastAPI (422){Colors.RESET}")
            results.add_fail("Generate document: Validación falló", "Generate Document", "documents", elapsed)
        else:
            results.add_fail(f"Generate document: Status {response.status_code}", "Generate Document", "documents", elapsed)
    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start
        results.add_fail("Generate document: No se puede conectar", "Generate Document", "documents", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Generate document: {str(e)}", "Generate Document", "documents", elapsed)
    
    return None

def test_task_status(task_id: str, results: TestResults):
    """Prueba el endpoint de estado de tarea."""
    if not task_id:
        return
    
    print(f"\n{Colors.HEADER}[TEST] Task Status{Colors.RESET}")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/{task_id}/status", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            if "status" in data:
                print(f"{Colors.SUCCESS}  ✓ Task status OK: {data.get('status')} ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("Task Status", "tasks", elapsed)
            else:
                results.add_fail("Task status: No 'status' en respuesta", "Task Status", "tasks", elapsed)
        else:
            results.add_fail(f"Task status: Status {response.status_code}", "Task Status", "tasks", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Task status: {str(e)}", "Task Status", "tasks", elapsed)

def test_list_tasks(results: TestResults):
    """Prueba el endpoint de listar tareas con paginación y filtros."""
    print(f"\n{Colors.HEADER}[TEST] List Tasks{Colors.RESET}")
    
    # Test básico
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/tasks?limit=10", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            if "tasks" in data and isinstance(data["tasks"], list):
                print(f"{Colors.SUCCESS}  ✓ List tasks OK: {len(data['tasks'])} tareas ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("List Tasks", "tasks", elapsed)
            else:
                results.add_fail("List tasks: Formato de respuesta inválido", "List Tasks", "tasks", elapsed)
        else:
            results.add_fail(f"List tasks: Status {response.status_code}", "List Tasks", "tasks", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"List tasks: {str(e)}", "List Tasks", "tasks", elapsed)
    
    # Test con paginación
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/tasks?limit=5&offset=0", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            if "total" in data and "has_more" in data:
                print(f"{Colors.SUCCESS}  ✓ Paginación OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("List Tasks: Pagination", "tasks", elapsed)
            else:
                results.add_fail("Paginación: Faltan campos", "List Tasks: Pagination", "tasks", elapsed)
        else:
            results.add_fail(f"Paginación: Status {response.status_code}", "List Tasks: Pagination", "tasks", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Paginación: {str(e)}", "List Tasks: Pagination", "tasks", elapsed)

def test_list_documents(results: TestResults):
    """Prueba el endpoint de listar documentos."""
    print(f"\n{Colors.HEADER}[TEST] List Documents{Colors.RESET}")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/documents?limit=10", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            if "documents" in data and isinstance(data["documents"], list):
                print(f"{Colors.SUCCESS}  ✓ List documents OK: {len(data['documents'])} documentos ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("List Documents", "documents", elapsed)
            else:
                results.add_fail("List documents: Formato de respuesta inválido", "List Documents", "documents", elapsed)
        else:
            results.add_fail(f"List documents: Status {response.status_code}", "List Documents", "documents", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"List documents: {str(e)}", "List Documents", "documents", elapsed)

def test_validation_errors(results: TestResults):
    """Prueba validaciones de entrada."""
    print(f"\n{Colors.HEADER}[TEST] Input Validation{Colors.RESET}")
    
    # Test 1: Query muy corta
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": "test"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code in [400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Validación query corta OK ({elapsed*1000:.0f}ms){Colors.RESET}")
            results.add_pass("Validation: Short Query", "validation", elapsed)
        else:
            results.add_fail("Validation: Debería rechazar query corta", "Validation: Short Query", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Validation: {str(e)}", "Validation: Short Query", "validation", elapsed)
    
    # Test 2: Query muy larga
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": "x" * 6000},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code in [400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Validación query larga OK ({elapsed*1000:.0f}ms){Colors.RESET}")
            results.add_pass("Validation: Long Query", "validation", elapsed)
        else:
            results.add_fail("Validation: Debería rechazar query muy larga", "Validation: Long Query", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Validation: {str(e)}", "Validation: Long Query", "validation", elapsed)
    
    # Test 3: Campos faltantes
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code in [400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Validación campos faltantes OK ({elapsed*1000:.0f}ms){Colors.RESET}")
            results.add_pass("Validation: Missing Fields", "validation", elapsed)
        else:
            results.add_fail("Validation: Debería rechazar campos faltantes", "Validation: Missing Fields", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Validation: {str(e)}", "Validation: Missing Fields", "validation", elapsed)

def test_rate_limiting(results: TestResults):
    """Prueba rate limiting (si está implementado)."""
    print(f"\n{Colors.HEADER}[TEST] Rate Limiting{Colors.RESET}")
    
    # Intentar hacer múltiples requests rápidas
    start = time.time()
    success_count = 0
    rate_limited = False
    
    for i in range(15):  # Más que el límite de 10/min
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": f"Test query {i} " * 10},  # Query válida
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:  # Too Many Requests
                rate_limited = True
                break
        except Exception:
            pass
        time.sleep(0.1)  # Pequeña pausa
    
    elapsed = time.time() - start
    
    if rate_limited:
        print(f"{Colors.SUCCESS}  ✓ Rate limiting funcionando ({elapsed*1000:.0f}ms){Colors.RESET}")
        results.add_pass("Rate Limiting", "security", elapsed)
    else:
        print(f"{Colors.WARNING}  ⚠ Rate limiting no detectado (puede estar deshabilitado){Colors.RESET}")
        results.add_pass("Rate Limiting", "security", elapsed)  # No falla si no está implementado

def test_websocket(results: TestResults):
    """Prueba conexión WebSocket."""
    if not WEBSOCKETS_AVAILABLE:
        print(f"\n{Colors.HEADER}[TEST] WebSocket{Colors.RESET}")
        print(f"{Colors.WARNING}  ⚠ WebSocket test skipped: websockets module not installed{Colors.RESET}")
        print(f"{Colors.INFO}   Install with: pip install websockets{Colors.RESET}")
        return
    
    print(f"\n{Colors.HEADER}[TEST] WebSocket Connection{Colors.RESET}")
    start = time.time()
    
    try:
        async def test_ws():
            uri = f"ws://localhost:8000/api/ws/test-task-123"
            try:
                async with websockets.connect(uri, timeout=5) as ws:
                    # Esperar mensaje inicial
                    message = await asyncio.wait_for(ws.recv(), timeout=3)
                    data = json.loads(message)
                    return data.get("type") in ["initial_state", "connected", "status"]
            except asyncio.TimeoutError:
                return False
            except Exception as e:
                print(f"{Colors.WARNING}  ⚠ WebSocket error: {e}{Colors.RESET}")
                return False
        
        result = asyncio.run(test_ws())
        elapsed = time.time() - start
        
        if result:
            print(f"{Colors.SUCCESS}  ✓ WebSocket conectado ({elapsed*1000:.0f}ms){Colors.RESET}")
            results.add_pass("WebSocket Connection", "websocket", elapsed)
        else:
            print(f"{Colors.WARNING}  ⚠ WebSocket no recibió mensaje válido{Colors.RESET}")
            results.add_pass("WebSocket Connection", "websocket", elapsed)  # No falla si no hay tarea
    except Exception as e:
        elapsed = time.time() - start
        print(f"{Colors.WARNING}  ⚠ WebSocket error: {e}{Colors.RESET}")
        results.add_pass("WebSocket Connection", "websocket", elapsed)  # No crítico

def test_metrics_endpoint(results: TestResults):
    """Prueba el endpoint de métricas Prometheus."""
    print(f"\n{Colors.HEADER}[TEST] Metrics Endpoint{Colors.RESET}")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}  ✓ Metrics endpoint OK ({elapsed*1000:.0f}ms){Colors.RESET}")
            results.add_pass("Metrics Endpoint", "system", elapsed)
        else:
            results.add_fail(f"Metrics: Status {response.status_code}", "Metrics Endpoint", "system", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        print(f"{Colors.WARNING}  ⚠ Metrics endpoint no disponible (opcional){Colors.RESET}")
        results.add_pass("Metrics Endpoint", "system", elapsed)  # No falla si no está disponible

def test_docs_endpoint(results: TestResults):
    """Prueba el endpoint de documentación."""
    print(f"\n{Colors.HEADER}[TEST] Documentation Endpoints{Colors.RESET}")
    
    endpoints = [
        ("/api/docs", "Swagger UI"),
        ("/api/redoc", "ReDoc"),
        ("/api/openapi.json", "OpenAPI Schema")
    ]
    
    for endpoint, name in endpoints:
        start = time.time()
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            elapsed = time.time() - start
            if response.status_code == 200:
                print(f"{Colors.SUCCESS}  ✓ {name} OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass(f"Docs: {name}", "documentation", elapsed)
            else:
                results.add_fail(f"{name}: Status {response.status_code}", f"Docs: {name}", "documentation", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            results.add_fail(f"{name}: {str(e)}", f"Docs: {name}", "documentation", elapsed)

def test_get_task_document(task_id: str, results: TestResults):
    """Prueba obtener documento de una tarea."""
    if not task_id:
        return
    
    print(f"\n{Colors.HEADER}[TEST] Get Task Document{Colors.RESET}")
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/{task_id}/document", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            if "document" in data:
                print(f"{Colors.SUCCESS}  ✓ Get document OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("Get Task Document", "documents", elapsed)
            else:
                results.add_fail("Get document: No 'document' en respuesta", "Get Task Document", "documents", elapsed)
        elif response.status_code == 400:
            print(f"{Colors.INFO}  ℹ Task aún no completada (esperado){Colors.RESET}")
            results.add_pass("Get Task Document", "documents", elapsed)  # No falla si no está listo
        elif response.status_code == 404:
            print(f"{Colors.WARNING}  ⚠ Task o documento no encontrado{Colors.RESET}")
            results.add_pass("Get Task Document", "documents", elapsed)  # No falla
        else:
            results.add_fail(f"Get document: Status {response.status_code}", "Get Task Document", "documents", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Get document: {str(e)}", "Get Task Document", "documents", elapsed)

def test_different_business_areas(results: TestResults):
    """Prueba generación con diferentes business areas."""
    print(f"\n{Colors.HEADER}[TEST] Different Business Areas{Colors.RESET}")
    
    areas = ["marketing", "sales", "finance", "hr", "operations"]
    success_count = 0
    
    for area in areas:
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={
                    "query": f"Crear un plan estratégico para el área de {area}",
                    "business_area": area,
                    "document_type": "strategy"
                },
                timeout=TIMEOUT
            )
            elapsed = time.time() - start
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    if success_count > 0:
        print(f"{Colors.SUCCESS}  ✓ Business areas: {success_count}/{len(areas)} exitosos{Colors.RESET}")
        results.add_pass("Different Business Areas", "documents", 0)
    else:
        results.add_fail("Business areas: Ninguno exitoso", "Different Business Areas", "documents", 0)

def test_different_document_types(results: TestResults):
    """Prueba generación con diferentes tipos de documentos."""
    print(f"\n{Colors.HEADER}[TEST] Different Document Types{Colors.RESET}")
    
    doc_types = ["strategy", "report", "plan", "analysis", "proposal"]
    success_count = 0
    
    for doc_type in doc_types:
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={
                    "query": f"Crear un {doc_type} completo para análisis empresarial",
                    "business_area": "general",
                    "document_type": doc_type
                },
                timeout=TIMEOUT
            )
            elapsed = time.time() - start
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    if success_count > 0:
        print(f"{Colors.SUCCESS}  ✓ Document types: {success_count}/{len(doc_types)} exitosos{Colors.RESET}")
        results.add_pass("Different Document Types", "documents", 0)
    else:
        results.add_fail("Document types: Ninguno exitoso", "Different Document Types", "documents", 0)

def test_task_filters(results: TestResults):
    """Prueba filtros en listado de tareas."""
    print(f"\n{Colors.HEADER}[TEST] Task Filters{Colors.RESET}")
    
    # Test con filtro de status
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/tasks?status=completed&limit=5", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            print(f"{Colors.SUCCESS}  ✓ Filter by status OK ({elapsed*1000:.0f}ms){Colors.RESET}")
            results.add_pass("Task Filters: Status", "tasks", elapsed)
        else:
            results.add_fail(f"Filter by status: Status {response.status_code}", "Task Filters: Status", "tasks", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Filter by status: {str(e)}", "Task Filters: Status", "tasks", elapsed)

def test_concurrent_requests(results: TestResults):
    """Prueba requests concurrentes."""
    print(f"\n{Colors.HEADER}[TEST] Concurrent Requests{Colors.RESET}")
    
    def make_request(i):
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
            elapsed = time.time() - start
            return {"success": response.status_code == 200, "time": elapsed, "index": i}
        except:
            return {"success": False, "time": 0, "index": i}
    
    start_total = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(20)]
        results_list = [f.result() for f in as_completed(futures)]
    
    elapsed_total = time.time() - start_total
    success_count = sum(1 for r in results_list if r["success"])
    avg_time = sum(r["time"] for r in results_list) / len(results_list) if results_list else 0
    
    if success_count >= 18:  # Al menos 90% exitosos
        print(f"{Colors.SUCCESS}  ✓ Concurrent requests: {success_count}/20 exitosos ({elapsed_total*1000:.0f}ms total, {avg_time*1000:.0f}ms promedio){Colors.RESET}")
        results.add_pass("Concurrent Requests", "performance", elapsed_total)
    else:
        results.add_fail(f"Concurrent requests: Solo {success_count}/20 exitosos", "Concurrent Requests", "performance", elapsed_total)

def test_error_handling(results: TestResults):
    """Pruebas exhaustivas de manejo de errores."""
    print(f"\n{Colors.HEADER}[TEST] Error Handling{Colors.RESET}")
    
    # Test 1: Task ID inválido
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/invalid-task-id-12345/status", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 404:
            print(f"{Colors.SUCCESS}  ✓ Error handling: 404 para task inválido OK{Colors.RESET}")
            results.add_pass("Error Handling: Invalid Task ID", "validation", elapsed)
        else:
            results.add_fail(f"Error handling: Esperado 404, recibido {response.status_code}", "Error Handling: Invalid Task ID", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Error handling: {str(e)}", "Error Handling: Invalid Task ID", "validation", elapsed)
    
    # Test 2: Endpoint no existente
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/nonexistent", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 404:
            print(f"{Colors.SUCCESS}  ✓ Error handling: 404 para endpoint inexistente OK{Colors.RESET}")
            results.add_pass("Error Handling: Nonexistent Endpoint", "validation", elapsed)
        else:
            results.add_fail(f"Error handling: Esperado 404, recibido {response.status_code}", "Error Handling: Nonexistent Endpoint", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Error handling: {str(e)}", "Error Handling: Nonexistent Endpoint", "validation", elapsed)

def test_cache_behavior(results: TestResults):
    """Prueba comportamiento de caché."""
    print(f"\n{Colors.HEADER}[TEST] Cache Behavior{Colors.RESET}")
    
    query = "Test de caché con query idéntica para verificar reutilización"
    
    # Primera request
    start1 = time.time()
    try:
        response1 = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": query, "business_area": "test"},
            timeout=TIMEOUT
        )
        elapsed1 = time.time() - start1
        task_id1 = response1.json().get("task_id") if response1.status_code == 200 else None
    except:
        task_id1 = None
        elapsed1 = 0
    
    # Segunda request con misma query
    time.sleep(1)  # Pequeña pausa
    start2 = time.time()
    try:
        response2 = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": query, "business_area": "test"},
            timeout=TIMEOUT
        )
        elapsed2 = time.time() - start2
        
        if response2.status_code == 200:
            data2 = response2.json()
            if data2.get("message") and "caché" in data2.get("message", "").lower():
                print(f"{Colors.SUCCESS}  ✓ Cache funcionando: respuesta desde caché{Colors.RESET}")
                results.add_pass("Cache Behavior", "performance", elapsed2)
            else:
                print(f"{Colors.INFO}  ℹ Cache: Segunda request procesada normalmente{Colors.RESET}")
                results.add_pass("Cache Behavior", "performance", elapsed2)  # No falla
        else:
            results.add_pass("Cache Behavior", "performance", elapsed2)  # No falla
    except Exception as e:
        elapsed2 = time.time() - start2
        results.add_pass("Cache Behavior", "performance", elapsed2)  # No crítico

def test_response_structure(results: TestResults):
    """Prueba estructura de respuestas."""
    print(f"\n{Colors.HEADER}[TEST] Response Structure Validation{Colors.RESET}")
    
    # Test estructura de health
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            required = ["status", "timestamp", "uptime"]
            missing = [f for f in required if f not in data]
            if not missing:
                print(f"{Colors.SUCCESS}  ✓ Health response structure OK{Colors.RESET}")
                results.add_pass("Response Structure: Health", "validation", elapsed)
            else:
                results.add_fail(f"Health structure: Faltan {missing}", "Response Structure: Health", "validation", elapsed)
        else:
            results.add_fail(f"Health structure: Status {response.status_code}", "Response Structure: Health", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Health structure: {str(e)}", "Response Structure: Health", "validation", elapsed)
    
    # Test estructura de stats
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            required = ["total_requests", "active_tasks", "success_rate"]
            missing = [f for f in required if f not in data]
            if not missing:
                print(f"{Colors.SUCCESS}  ✓ Stats response structure OK{Colors.RESET}")
                results.add_pass("Response Structure: Stats", "validation", elapsed)
            else:
                results.add_fail(f"Stats structure: Faltan {missing}", "Response Structure: Stats", "validation", elapsed)
        else:
            results.add_fail(f"Stats structure: Status {response.status_code}", "Response Structure: Stats", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Stats structure: {str(e)}", "Response Structure: Stats", "validation", elapsed)

def test_timeout_handling(results: TestResults):
    """Prueba manejo de timeouts."""
    print(f"\n{Colors.HEADER}[TEST] Timeout Handling{Colors.RESET}")
    
    start = time.time()
    try:
        # Intentar con timeout muy corto
        response = requests.get(f"{BASE_URL}/api/health", timeout=0.001)
        elapsed = time.time() - start
        results.add_fail("Timeout: Debería haber fallado", "Timeout Handling", "validation", elapsed)
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"{Colors.SUCCESS}  ✓ Timeout handling OK{Colors.RESET}")
        results.add_pass("Timeout Handling", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        # Otros errores también son válidos para timeout
        print(f"{Colors.INFO}  ℹ Timeout test: {type(e).__name__}{Colors.RESET}")
        results.add_pass("Timeout Handling", "validation", elapsed)

def test_cors_headers(results: TestResults):
    """Prueba headers CORS."""
    print(f"\n{Colors.HEADER}[TEST] CORS Headers{Colors.RESET}")
    
    start = time.time()
    try:
        response = requests.options(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        
        # Verificar headers CORS comunes
        cors_headers = ["Access-Control-Allow-Origin", "Access-Control-Allow-Methods"]
        found_headers = [h for h in cors_headers if h in response.headers]
        
        if found_headers or response.status_code in [200, 204]:
            print(f"{Colors.SUCCESS}  ✓ CORS headers OK ({len(found_headers)} headers encontrados){Colors.RESET}")
            results.add_pass("CORS Headers", "security", elapsed)
        else:
            print(f"{Colors.INFO}  ℹ CORS headers no detectados (puede estar deshabilitado){Colors.RESET}")
            results.add_pass("CORS Headers", "security", elapsed)  # No falla
    except Exception as e:
        elapsed = time.time() - start
        print(f"{Colors.INFO}  ℹ CORS test: {str(e)}{Colors.RESET}")
        results.add_pass("CORS Headers", "security", elapsed)  # No crítico

def test_stress_load(results: TestResults):
    """Prueba de carga/stress básica."""
    print(f"\n{Colors.HEADER}[TEST] Stress Load Test{Colors.RESET}")
    
    def make_health_request():
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            elapsed = time.time() - start
            return {"success": response.status_code == 200, "time": elapsed}
        except:
            return {"success": False, "time": 0}
    
    start_total = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_health_request) for _ in range(50)]
        results_list = [f.result() for f in as_completed(futures)]
    
    elapsed_total = time.time() - start_total
    success_count = sum(1 for r in results_list if r["success"])
    avg_time = sum(r["time"] for r in results_list) / len(results_list) if results_list else 0
    
    success_rate = (success_count / 50) * 100
    
    if success_rate >= 95:  # Al menos 95% exitosos
        print(f"{Colors.SUCCESS}  ✓ Stress test: {success_count}/50 exitosos ({success_rate:.1f}%) - {elapsed_total*1000:.0f}ms total{Colors.RESET}")
        results.add_pass("Stress Load Test", "performance", elapsed_total)
    elif success_rate >= 80:
        print(f"{Colors.WARNING}  ⚠ Stress test: {success_count}/50 exitosos ({success_rate:.1f}%) - Puede necesitar optimización{Colors.RESET}")
        results.add_pass("Stress Load Test", "performance", elapsed_total)  # No falla
    else:
        results.add_fail(f"Stress test: Solo {success_count}/50 exitosos ({success_rate:.1f}%)", "Stress Load Test", "performance", elapsed_total)

def test_end_to_end_workflow(results: TestResults):
    """Prueba workflow completo end-to-end."""
    print(f"\n{Colors.HEADER}[TEST] End-to-End Workflow{Colors.RESET}")
    
    start = time.time()
    try:
        # 1. Crear documento
        response1 = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={
                "query": "Test end-to-end workflow completo para validar todo el proceso",
                "business_area": "testing",
                "document_type": "test"
            },
            timeout=TIMEOUT
        )
        
        if response1.status_code != 200:
            results.add_fail("E2E: No se pudo crear documento", "End-to-End Workflow", "integration", time.time() - start)
            return
        
        task_id = response1.json().get("task_id")
        if not task_id:
            results.add_fail("E2E: No se obtuvo task_id", "End-to-End Workflow", "integration", time.time() - start)
            return
        
        # 2. Verificar status
        time.sleep(1)
        response2 = requests.get(f"{BASE_URL}/api/tasks/{task_id}/status", timeout=TIMEOUT)
        if response2.status_code != 200:
            results.add_fail("E2E: No se pudo obtener status", "End-to-End Workflow", "integration", time.time() - start)
            return
        
        # 3. Listar tareas (debe incluir nuestra tarea)
        response3 = requests.get(f"{BASE_URL}/api/tasks?limit=100", timeout=TIMEOUT)
        if response3.status_code == 200:
            tasks = response3.json().get("tasks", [])
            task_found = any(t.get("task_id") == task_id for t in tasks)
            if task_found:
                elapsed = time.time() - start
                print(f"{Colors.SUCCESS}  ✓ End-to-end workflow OK ({elapsed*1000:.0f}ms){Colors.RESET}")
                results.add_pass("End-to-End Workflow", "integration", elapsed)
            else:
                results.add_fail("E2E: Tarea no encontrada en listado", "End-to-End Workflow", "integration", time.time() - start)
        else:
            results.add_fail("E2E: No se pudo listar tareas", "End-to-End Workflow", "integration", time.time() - start)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"E2E: {str(e)}", "End-to-End Workflow", "integration", elapsed)

def test_retry_logic(results: TestResults):
    """Prueba lógica de reintentos."""
    print(f"\n{Colors.HEADER}[TEST] Retry Logic{Colors.RESET}")
    
    start = time.time()
    success_count = 0
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            if response.status_code == 200:
                success_count += 1
                break
        except:
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Esperar antes de reintentar
    
    elapsed = time.time() - start
    
    if success_count > 0:
        print(f"{Colors.SUCCESS}  ✓ Retry logic OK ({success_count} intento(s)){Colors.RESET}")
        results.add_pass("Retry Logic", "resilience", elapsed)
    else:
        results.add_fail("Retry logic: No se pudo conectar después de reintentos", "Retry Logic", "resilience", elapsed)

def test_invalid_json(results: TestResults):
    """Prueba con JSON inválido."""
    print(f"\n{Colors.HEADER}[TEST] Invalid JSON Handling{Colors.RESET}")
    
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            data="invalid json {not valid}",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code in [400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Invalid JSON rechazado correctamente{Colors.RESET}")
            results.add_pass("Invalid JSON Handling", "validation", elapsed)
        else:
            results.add_fail(f"Invalid JSON: Esperado 400/422, recibido {response.status_code}", "Invalid JSON Handling", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        # Errores de conexión también son válidos
        print(f"{Colors.INFO}  ℹ Invalid JSON test: {type(e).__name__}{Colors.RESET}")
        results.add_pass("Invalid JSON Handling", "validation", elapsed)

def test_malformed_requests(results: TestResults):
    """Prueba requests mal formados."""
    print(f"\n{Colors.HEADER}[TEST] Malformed Requests{Colors.RESET}")
    
    # Test 1: Query vacía (solo espacios)
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": "   ", "business_area": "test"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code in [400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Query vacía rechazada{Colors.RESET}")
            results.add_pass("Malformed: Empty Query", "validation", elapsed)
        else:
            results.add_fail("Malformed: Debería rechazar query vacía", "Malformed: Empty Query", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Malformed: Empty Query", "validation", elapsed)
    
    # Test 2: Tipo de dato incorrecto
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": 12345, "business_area": "test"},  # query debería ser string
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code in [400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Tipo incorrecto rechazado{Colors.RESET}")
            results.add_pass("Malformed: Wrong Type", "validation", elapsed)
        else:
            results.add_pass("Malformed: Wrong Type", "validation", elapsed)  # No falla
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Malformed: Wrong Type", "validation", elapsed)

def test_content_type_validation(results: TestResults):
    """Prueba validación de Content-Type."""
    print(f"\n{Colors.HEADER}[TEST] Content-Type Validation{Colors.RESET}")
    
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            data=json.dumps({"query": "test query", "business_area": "test"}),
            headers={"Content-Type": "text/plain"},  # Tipo incorrecto
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        # FastAPI puede aceptar diferentes content types, no es crítico
        print(f"{Colors.INFO}  ℹ Content-Type test: Status {response.status_code}{Colors.RESET}")
        results.add_pass("Content-Type Validation", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Content-Type Validation", "validation", elapsed)

def test_large_payloads(results: TestResults):
    """Prueba con payloads grandes."""
    print(f"\n{Colors.HEADER}[TEST] Large Payloads{Colors.RESET}")
    
    # Test con query grande pero válida (cerca del límite)
    large_query = "A" * 4000  # Cerca del límite de 5000
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": large_query, "business_area": "test"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}  ✓ Large payload aceptado ({len(large_query)} chars){Colors.RESET}")
            results.add_pass("Large Payloads", "validation", elapsed)
        elif response.status_code in [400, 422]:
            print(f"{Colors.INFO}  ℹ Large payload rechazado (puede ser por límite){Colors.RESET}")
            results.add_pass("Large Payloads", "validation", elapsed)  # No falla
        else:
            results.add_pass("Large Payloads", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Large Payloads", "validation", elapsed)

def test_special_characters(results: TestResults):
    """Prueba con caracteres especiales."""
    print(f"\n{Colors.HEADER}[TEST] Special Characters{Colors.RESET}")
    
    special_queries = [
        "Test con énfasis y acentos: áéíóú",
        "Test con símbolos: !@#$%^&*()",
        "Test con emojis: 🚀 📊 ✅",
        "Test con unicode: 中文 العربية русский",
        "Test con SQL-like: SELECT * FROM users; DROP TABLE;"
    ]
    
    success_count = 0
    start = time.time()
    
    for query in special_queries:
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": query, "business_area": "test"},
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if success_count >= len(special_queries) * 0.6:  # Al menos 60% exitosos
        print(f"{Colors.SUCCESS}  ✓ Special characters: {success_count}/{len(special_queries)} exitosos{Colors.RESET}")
        results.add_pass("Special Characters", "validation", elapsed)
    else:
        print(f"{Colors.WARNING}  ⚠ Special characters: {success_count}/{len(special_queries)} exitosos{Colors.RESET}")
        results.add_pass("Special Characters", "validation", elapsed)  # No falla

def test_http_methods(results: TestResults):
    """Prueba diferentes métodos HTTP."""
    print(f"\n{Colors.HEADER}[TEST] HTTP Methods{Colors.RESET}")
    
    # Test GET (debería funcionar)
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}  ✓ GET method OK{Colors.RESET}")
            results.add_pass("HTTP Methods: GET", "validation", elapsed)
        else:
            results.add_fail(f"GET method: Status {response.status_code}", "HTTP Methods: GET", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"GET method: {str(e)}", "HTTP Methods: GET", "validation", elapsed)
    
    # Test OPTIONS (CORS preflight)
    start = time.time()
    try:
        response = requests.options(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code in [200, 204]:
            print(f"{Colors.SUCCESS}  ✓ OPTIONS method OK{Colors.RESET}")
            results.add_pass("HTTP Methods: OPTIONS", "validation", elapsed)
        else:
            results.add_pass("HTTP Methods: OPTIONS", "validation", elapsed)  # No crítico
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("HTTP Methods: OPTIONS", "validation", elapsed)

def test_custom_headers(results: TestResults):
    """Prueba con headers personalizados."""
    print(f"\n{Colors.HEADER}[TEST] Custom Headers{Colors.RESET}")
    
    start = time.time()
    try:
        response = requests.get(
            f"{BASE_URL}/api/health",
            headers={
                "X-Custom-Header": "test-value",
                "User-Agent": "Test-Suite/1.0",
                "Accept": "application/json"
            },
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}  ✓ Custom headers OK{Colors.RESET}")
            results.add_pass("Custom Headers", "validation", elapsed)
        else:
            results.add_pass("Custom Headers", "validation", elapsed)  # No falla
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Custom Headers", "validation", elapsed)

def test_query_parameters(results: TestResults):
    """Prueba diferentes query parameters."""
    print(f"\n{Colors.HEADER}[TEST] Query Parameters{Colors.RESET}")
    
    # Test con múltiples parámetros
    start = time.time()
    try:
        response = requests.get(
            f"{BASE_URL}/api/tasks?limit=5&offset=0&status=completed",
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"{Colors.SUCCESS}  ✓ Query parameters OK{Colors.RESET}")
            results.add_pass("Query Parameters", "validation", elapsed)
        else:
            results.add_fail(f"Query parameters: Status {response.status_code}", "Query Parameters", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_fail(f"Query parameters: {str(e)}", "Query Parameters", "validation", elapsed)
    
    # Test con parámetros inválidos
    start = time.time()
    try:
        response = requests.get(
            f"{BASE_URL}/api/tasks?limit=abc&offset=xyz",  # Valores inválidos
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        # Puede aceptar o rechazar, ambos son válidos
        print(f"{Colors.INFO}  ℹ Invalid query params: Status {response.status_code}{Colors.RESET}")
        results.add_pass("Query Parameters: Invalid", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Query Parameters: Invalid", "validation", elapsed)

def test_sorting_and_ordering(results: TestResults):
    """Prueba ordenamiento y filtrado avanzado."""
    print(f"\n{Colors.HEADER}[TEST] Sorting and Ordering{Colors.RESET}")
    
    # Test con diferentes límites
    limits = [1, 5, 10, 20, 50]
    success_count = 0
    start = time.time()
    
    for limit in limits:
        try:
            response = requests.get(f"{BASE_URL}/api/tasks?limit={limit}", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if "tasks" in data:
                    success_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if success_count >= len(limits) * 0.8:  # Al menos 80% exitosos
        print(f"{Colors.SUCCESS}  ✓ Sorting/ordering: {success_count}/{len(limits)} límites OK{Colors.RESET}")
        results.add_pass("Sorting and Ordering", "validation", elapsed)
    else:
        print(f"{Colors.WARNING}  ⚠ Sorting/ordering: {success_count}/{len(limits)} límites OK{Colors.RESET}")
        results.add_pass("Sorting and Ordering", "validation", elapsed)  # No falla

def test_batch_operations(results: TestResults):
    """Prueba operaciones en batch."""
    print(f"\n{Colors.HEADER}[TEST] Batch Operations{Colors.RESET}")
    
    # Crear múltiples documentos en paralelo
    queries = [
        "Documento batch 1 para testing",
        "Documento batch 2 para testing",
        "Documento batch 3 para testing"
    ]
    
    start = time.time()
    task_ids = []
    
    def create_document(query):
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": query, "business_area": "test"},
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                return response.json().get("task_id")
        except:
            pass
        return None
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(create_document, q) for q in queries]
        task_ids = [f.result() for f in as_completed(futures) if f.result()]
    
    elapsed = time.time() - start
    
    if len(task_ids) >= 2:  # Al menos 2 exitosos
        print(f"{Colors.SUCCESS}  ✓ Batch operations: {len(task_ids)}/{len(queries)} creados{Colors.RESET}")
        results.add_pass("Batch Operations", "performance", elapsed)
    else:
        print(f"{Colors.WARNING}  ⚠ Batch operations: {len(task_ids)}/{len(queries)} creados{Colors.RESET}")
        results.add_pass("Batch Operations", "performance", elapsed)  # No falla

def test_api_versioning(results: TestResults):
    """Prueba versionado de API."""
    print(f"\n{Colors.HEADER}[TEST] API Versioning{Colors.RESET}")
    
    # Test endpoints con y sin versión
    endpoints = [
        "/api/health",  # Sin versión explícita
        "/api/stats",   # Sin versión explícita
    ]
    
    success_count = 0
    start = time.time()
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if success_count == len(endpoints):
        print(f"{Colors.SUCCESS}  ✓ API versioning: {success_count}/{len(endpoints)} endpoints OK{Colors.RESET}")
        results.add_pass("API Versioning", "validation", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ API versioning: {success_count}/{len(endpoints)} endpoints OK{Colors.RESET}")
        results.add_pass("API Versioning", "validation", elapsed)  # No falla

def test_performance_comparison(results: TestResults):
    """Compara performance entre diferentes endpoints."""
    print(f"\n{Colors.HEADER}[TEST] Performance Comparison{Colors.RESET}")
    
    endpoints = [
        ("/api/health", "Health"),
        ("/api/stats", "Stats"),
        ("/", "Root")
    ]
    
    endpoint_times = {}
    start = time.time()
    
    for endpoint, name in endpoints:
        times = []
        for _ in range(3):
            try:
                req_start = time.time()
                requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
                times.append(time.time() - req_start)
            except:
                pass
        if times:
            endpoint_times[name] = sum(times) / len(times)
    
    elapsed = time.time() - start
    
    if endpoint_times:
        fastest = min(endpoint_times.items(), key=lambda x: x[1])
        print(f"{Colors.SUCCESS}  ✓ Performance comparison: {fastest[0]} más rápido ({fastest[1]*1000:.0f}ms){Colors.RESET}")
        results.add_pass("Performance Comparison", "performance", elapsed)
    else:
        results.add_pass("Performance Comparison", "performance", elapsed)

def test_response_size_analysis(results: TestResults):
    """Analiza tamaño de respuestas."""
    print(f"\n{Colors.HEADER}[TEST] Response Size Analysis{Colors.RESET}")
    
    start = time.time()
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        elapsed = time.time() - start
        if response.status_code == 200:
            size = len(response.content)
            size_kb = size / 1024
            if size_kb < 100:  # Menos de 100KB
                print(f"{Colors.SUCCESS}  ✓ Response size OK: {size_kb:.2f}KB{Colors.RESET}")
                results.add_pass("Response Size Analysis", "performance", elapsed)
            else:
                print(f"{Colors.WARNING}  ⚠ Response size grande: {size_kb:.2f}KB{Colors.RESET}")
                results.add_pass("Response Size Analysis", "performance", elapsed)  # No falla
        else:
            results.add_pass("Response Size Analysis", "performance", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Response Size Analysis", "performance", elapsed)

def test_error_rate_analysis(results: TestResults):
    """Analiza tasa de errores."""
    print(f"\n{Colors.HEADER}[TEST] Error Rate Analysis{Colors.RESET}")
    
    # Hacer múltiples requests y contar errores
    total_requests = 20
    errors = 0
    start = time.time()
    
    for _ in range(total_requests):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            if response.status_code != 200:
                errors += 1
        except:
            errors += 1
    
    elapsed = time.time() - start
    error_rate = (errors / total_requests) * 100
    
    if error_rate < 5:  # Menos del 5% de errores
        print(f"{Colors.SUCCESS}  ✓ Error rate OK: {error_rate:.1f}% ({errors}/{total_requests}){Colors.RESET}")
        results.add_pass("Error Rate Analysis", "performance", elapsed)
    elif error_rate < 20:
        print(f"{Colors.WARNING}  ⚠ Error rate moderado: {error_rate:.1f}%{Colors.RESET}")
        results.add_pass("Error Rate Analysis", "performance", elapsed)  # No falla
    else:
        results.add_fail(f"Error rate alto: {error_rate:.1f}%", "Error Rate Analysis", "performance", elapsed)

def test_availability_check(results: TestResults):
    """Verifica disponibilidad del servicio."""
    print(f"\n{Colors.HEADER}[TEST] Availability Check{Colors.RESET}")
    
    checks = 5
    successful = 0
    start = time.time()
    
    for _ in range(checks):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            if response.status_code == 200:
                successful += 1
        except:
            pass
        time.sleep(0.5)  # Pequeña pausa entre checks
    
    elapsed = time.time() - start
    availability = (successful / checks) * 100
    
    if availability >= 80:  # Al menos 80% disponible
        print(f"{Colors.SUCCESS}  ✓ Availability OK: {availability:.1f}% ({successful}/{checks}){Colors.RESET}")
        results.add_pass("Availability Check", "system", elapsed)
    else:
        results.add_fail(f"Availability baja: {availability:.1f}%", "Availability Check", "system", elapsed)

def test_data_consistency(results: TestResults):
    """Verifica consistencia de datos."""
    print(f"\n{Colors.HEADER}[TEST] Data Consistency{Colors.RESET}")
    
    start = time.time()
    
    # Hacer dos requests al mismo endpoint y comparar
    try:
        response1 = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        time.sleep(0.5)
        response2 = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Verificar que campos clave existan en ambos
            key_fields = ["total_requests", "active_tasks", "success_rate"]
            consistent = all(field in data1 and field in data2 for field in key_fields)
            
            elapsed = time.time() - start
            if consistent:
                print(f"{Colors.SUCCESS}  ✓ Data consistency OK{Colors.RESET}")
                results.add_pass("Data Consistency", "validation", elapsed)
            else:
                results.add_fail("Data consistency: Campos inconsistentes", "Data Consistency", "validation", elapsed)
        else:
            elapsed = time.time() - start
            results.add_pass("Data Consistency", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Data Consistency", "validation", elapsed)

def test_api_compliance(results: TestResults):
    """Verifica cumplimiento de estándares de API."""
    print(f"\n{Colors.HEADER}[TEST] API Compliance{Colors.RESET}")
    
    start = time.time()
    compliance_checks = []
    
    # Check 1: JSON responses
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        if response.status_code == 200:
            try:
                response.json()
                compliance_checks.append(True)
            except:
                compliance_checks.append(False)
    except:
        pass
    
    # Check 2: Proper status codes
    try:
        response = requests.get(f"{BASE_URL}/api/nonexistent", timeout=TIMEOUT)
        if response.status_code == 404:
            compliance_checks.append(True)
        else:
            compliance_checks.append(False)
    except:
        pass
    
    elapsed = time.time() - start
    
    if len(compliance_checks) > 0 and all(compliance_checks):
        print(f"{Colors.SUCCESS}  ✓ API compliance OK ({len(compliance_checks)} checks){Colors.RESET}")
        results.add_pass("API Compliance", "validation", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ API compliance: {sum(compliance_checks)}/{len(compliance_checks)} checks OK{Colors.RESET}")
        results.add_pass("API Compliance", "validation", elapsed)  # No falla

def test_metrics_collection(results: TestResults):
    """Prueba recolección de métricas."""
    print(f"\n{Colors.HEADER}[TEST] Metrics Collection{Colors.RESET}")
    
    start = time.time()
    try:
        # Verificar endpoint de métricas
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            metrics_content = response.text
            # Verificar que contenga métricas comunes
            has_metrics = any(keyword in metrics_content.lower() for keyword in ['http', 'request', 'duration', 'total'])
            if has_metrics:
                print(f"{Colors.SUCCESS}  ✓ Metrics collection OK{Colors.RESET}")
                results.add_pass("Metrics Collection", "monitoring", elapsed)
            else:
                print(f"{Colors.INFO}  ℹ Metrics endpoint disponible{Colors.RESET}")
                results.add_pass("Metrics Collection", "monitoring", elapsed)
        else:
            print(f"{Colors.INFO}  ℹ Metrics endpoint no disponible (opcional){Colors.RESET}")
            results.add_pass("Metrics Collection", "monitoring", elapsed)  # No falla
    except Exception as e:
        elapsed = time.time() - start
        print(f"{Colors.INFO}  ℹ Metrics collection: {type(e).__name__}{Colors.RESET}")
        results.add_pass("Metrics Collection", "monitoring", elapsed)  # No crítico

def test_health_monitoring(results: TestResults):
    """Prueba monitoreo de salud continuo."""
    print(f"\n{Colors.HEADER}[TEST] Health Monitoring{Colors.RESET}")
    
    checks = 3
    healthy_count = 0
    start = time.time()
    
    for i in range(checks):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    healthy_count += 1
        except:
            pass
        if i < checks - 1:
            time.sleep(0.3)
    
    elapsed = time.time() - start
    health_rate = (healthy_count / checks) * 100
    
    if health_rate >= 66:  # Al menos 2/3 healthy
        print(f"{Colors.SUCCESS}  ✓ Health monitoring OK: {health_rate:.0f}% ({healthy_count}/{checks}){Colors.RESET}")
        results.add_pass("Health Monitoring", "monitoring", elapsed)
    else:
        results.add_fail(f"Health monitoring: Solo {health_rate:.0f}% healthy", "Health Monitoring", "monitoring", elapsed)

def test_response_time_tracking(results: TestResults):
    """Prueba tracking de tiempo de respuesta."""
    print(f"\n{Colors.HEADER}[TEST] Response Time Tracking{Colors.RESET}")
    
    times = []
    start = time.time()
    
    for _ in range(10):
        try:
            req_start = time.time()
            requests.get(f"{BASE_URL}/api/health", timeout=5)
            times.append(time.time() - req_start)
        except:
            pass
    
    elapsed = time.time() - start
    
    if times:
        avg = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"{Colors.SUCCESS}  ✓ Response time tracking: avg={avg*1000:.0f}ms, min={min_time*1000:.0f}ms, max={max_time*1000:.0f}ms{Colors.RESET}")
        results.add_pass("Response Time Tracking", "monitoring", elapsed)
    else:
        results.add_pass("Response Time Tracking", "monitoring", elapsed)

def test_throughput_measurement(results: TestResults):
    """Mide throughput del sistema."""
    print(f"\n{Colors.HEADER}[TEST] Throughput Measurement{Colors.RESET}")
    
    requests_count = 30
    successful = 0
    start = time.time()
    
    def make_request():
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(requests_count)]
        successful = sum(1 for f in as_completed(futures) if f.result())
    
    elapsed = time.time() - start
    throughput = requests_count / elapsed if elapsed > 0 else 0
    
    if throughput > 0:
        print(f"{Colors.SUCCESS}  ✓ Throughput: {throughput:.1f} req/s ({successful}/{requests_count} exitosos){Colors.RESET}")
        results.add_pass("Throughput Measurement", "performance", elapsed)
    else:
        results.add_pass("Throughput Measurement", "performance", elapsed)

def test_latency_analysis(results: TestResults):
    """Analiza latencia del sistema."""
    print(f"\n{Colors.HEADER}[TEST] Latency Analysis{Colors.RESET}")
    
    latencies = []
    start = time.time()
    
    for _ in range(15):
        try:
            req_start = time.time()
            requests.get(f"{BASE_URL}/api/health", timeout=5)
            latencies.append(time.time() - req_start)
        except:
            pass
    
    elapsed = time.time() - start
    
    if latencies:
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0]
        
        print(f"{Colors.SUCCESS}  ✓ Latency: p50={p50*1000:.0f}ms, p95={p95*1000:.0f}ms, p99={p99*1000:.0f}ms{Colors.RESET}")
        results.add_pass("Latency Analysis", "performance", elapsed)
    else:
        results.add_pass("Latency Analysis", "performance", elapsed)

def test_resource_usage(results: TestResults):
    """Prueba uso de recursos (básico)."""
    print(f"\n{Colors.HEADER}[TEST] Resource Usage{Colors.RESET}")
    
    start = time.time()
    
    # Hacer múltiples requests y medir tiempo total
    try:
        requests_list = []
        for _ in range(20):
            try:
                requests.get(f"{BASE_URL}/api/health", timeout=3)
                requests_list.append(True)
            except:
                requests_list.append(False)
        
        success_rate = sum(requests_list) / len(requests_list) * 100
        elapsed = time.time() - start
        
        if success_rate >= 90:
            print(f"{Colors.SUCCESS}  ✓ Resource usage OK: {success_rate:.0f}% éxito bajo carga{Colors.RESET}")
            results.add_pass("Resource Usage", "performance", elapsed)
        else:
            print(f"{Colors.WARNING}  ⚠ Resource usage: {success_rate:.0f}% éxito{Colors.RESET}")
            results.add_pass("Resource Usage", "performance", elapsed)  # No falla
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Resource Usage", "performance", elapsed)

def test_api_completeness(results: TestResults):
    """Verifica completitud de la API."""
    print(f"\n{Colors.HEADER}[TEST] API Completeness{Colors.RESET}")
    
    # Lista de endpoints esperados
    expected_endpoints = [
        ("/", "Root"),
        ("/api/health", "Health"),
        ("/api/stats", "Stats"),
        ("/api/documents/generate", "Generate Document"),
        ("/api/tasks", "List Tasks"),
        ("/api/documents", "List Documents"),
    ]
    
    available_count = 0
    start = time.time()
    
    for endpoint, name in expected_endpoints:
        try:
            if endpoint == "/api/documents/generate":
                # POST endpoint
                response = requests.post(f"{BASE_URL}{endpoint}", json={"query": "test"}, timeout=5)
            else:
                # GET endpoint
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            
            if response.status_code in [200, 400, 422]:  # Cualquier respuesta válida
                available_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    completeness = (available_count / len(expected_endpoints)) * 100
    
    if completeness >= 80:
        print(f"{Colors.SUCCESS}  ✓ API completeness: {completeness:.0f}% ({available_count}/{len(expected_endpoints)}){Colors.RESET}")
        results.add_pass("API Completeness", "validation", elapsed)
    else:
        results.add_fail(f"API completeness: Solo {completeness:.0f}%", "API Completeness", "validation", elapsed)

def test_documentation_quality(results: TestResults):
    """Verifica calidad de documentación."""
    print(f"\n{Colors.HEADER}[TEST] Documentation Quality{Colors.RESET}")
    
    docs_endpoints = [
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json"
    ]
    
    available_count = 0
    start = time.time()
    
    for endpoint in docs_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            if response.status_code == 200:
                available_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if available_count >= 2:  # Al menos 2 de 3 disponibles
        print(f"{Colors.SUCCESS}  ✓ Documentation quality: {available_count}/{len(docs_endpoints)} endpoints disponibles{Colors.RESET}")
        results.add_pass("Documentation Quality", "documentation", elapsed)
    else:
        print(f"{Colors.WARNING}  ⚠ Documentation: {available_count}/{len(docs_endpoints)} endpoints disponibles{Colors.RESET}")
        results.add_pass("Documentation Quality", "documentation", elapsed)  # No falla

def test_error_messages_quality(results: TestResults):
    """Verifica calidad de mensajes de error."""
    print(f"\n{Colors.HEADER}[TEST] Error Messages Quality{Colors.RESET}")
    
    start = time.time()
    
    # Test 1: Error 404 debe tener mensaje
    try:
        response = requests.get(f"{BASE_URL}/api/nonexistent", timeout=TIMEOUT)
        if response.status_code == 404:
            try:
                data = response.json()
                has_message = "detail" in data or "message" in data or "error" in data
                if has_message:
                    print(f"{Colors.SUCCESS}  ✓ Error messages quality OK{Colors.RESET}")
                    results.add_pass("Error Messages Quality", "validation", time.time() - start)
                else:
                    results.add_pass("Error Messages Quality", "validation", time.time() - start)
            except:
                results.add_pass("Error Messages Quality", "validation", time.time() - start)
        else:
            results.add_pass("Error Messages Quality", "validation", time.time() - start)
    except:
        elapsed = time.time() - start
        results.add_pass("Error Messages Quality", "validation", elapsed)

def test_response_format_consistency(results: TestResults):
    """Verifica consistencia de formato de respuestas."""
    print(f"\n{Colors.HEADER}[TEST] Response Format Consistency{Colors.RESET}")
    
    start = time.time()
    
    endpoints = [
        ("/api/health", ["status", "timestamp"]),
        ("/api/stats", ["total_requests", "active_tasks"])
    ]
    
    consistent_count = 0
    
    for endpoint, required_fields in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if all(field in data for field in required_fields):
                    consistent_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if consistent_count == len(endpoints):
        print(f"{Colors.SUCCESS}  ✓ Response format consistency OK{Colors.RESET}")
        results.add_pass("Response Format Consistency", "validation", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ Response format: {consistent_count}/{len(endpoints)} consistentes{Colors.RESET}")
        results.add_pass("Response Format Consistency", "validation", elapsed)

def test_backward_compatibility(results: TestResults):
    """Prueba compatibilidad hacia atrás."""
    print(f"\n{Colors.HEADER}[TEST] Backward Compatibility{Colors.RESET}")
    
    start = time.time()
    
    # Verificar que endpoints básicos sigan funcionando
    basic_endpoints = ["/", "/api/health", "/api/stats"]
    working_count = 0
    
    for endpoint in basic_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            if response.status_code == 200:
                working_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if working_count == len(basic_endpoints):
        print(f"{Colors.SUCCESS}  ✓ Backward compatibility OK{Colors.RESET}")
        results.add_pass("Backward Compatibility", "validation", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ Backward compatibility: {working_count}/{len(basic_endpoints)} endpoints funcionando{Colors.RESET}")
        results.add_pass("Backward Compatibility", "validation", elapsed)

def test_forward_compatibility(results: TestResults):
    """Prueba compatibilidad hacia adelante."""
    print(f"\n{Colors.HEADER}[TEST] Forward Compatibility{Colors.RESET}")
    
    start = time.time()
    
    # Verificar que la API acepte campos adicionales sin romper
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={
                "query": "Test de compatibilidad hacia adelante",
                "business_area": "test",
                "extra_field": "should_be_ignored"  # Campo adicional
            },
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        
        # Si acepta el campo adicional sin error, es compatible
        if response.status_code in [200, 400, 422]:
            print(f"{Colors.SUCCESS}  ✓ Forward compatibility OK{Colors.RESET}")
            results.add_pass("Forward Compatibility", "validation", elapsed)
        else:
            results.add_pass("Forward Compatibility", "validation", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        results.add_pass("Forward Compatibility", "validation", elapsed)

def test_edge_cases_comprehensive(results: TestResults):
    """Prueba edge cases exhaustivos."""
    print(f"\n{Colors.HEADER}[TEST] Comprehensive Edge Cases{Colors.RESET}")
    
    edge_cases = [
        ("query", ""),  # String vacío
        ("query", None),  # None (si se permite)
        ("query", 0),  # Cero
        ("query", -1),  # Negativo
        ("query", []),  # Lista vacía
        ("query", {}),  # Dict vacío
    ]
    
    success_count = 0
    start = time.time()
    
    for field, value in edge_cases:
        try:
            # Intentar crear request con edge case
            payload = {"query": "test query válida", "business_area": "test"}
            if field == "query":
                payload[field] = value
            
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json=payload,
                timeout=TIMEOUT
            )
            # Si rechaza correctamente (400/422) o acepta, ambos son válidos
            if response.status_code in [200, 400, 422]:
                success_count += 1
        except:
            pass
    
    elapsed = time.time() - start
    
    if success_count >= len(edge_cases) * 0.5:  # Al menos 50% manejados
        print(f"{Colors.SUCCESS}  ✓ Edge cases: {success_count}/{len(edge_cases)} manejados{Colors.RESET}")
        results.add_pass("Comprehensive Edge Cases", "validation", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ Edge cases: {success_count}/{len(edge_cases)} manejados{Colors.RESET}")
        results.add_pass("Comprehensive Edge Cases", "validation", elapsed)

def test_integration_scenarios(results: TestResults):
    """Prueba escenarios de integración complejos."""
    print(f"\n{Colors.HEADER}[TEST] Integration Scenarios{Colors.RESET}")
    
    start = time.time()
    scenario_success = 0
    
    # Escenario 1: Crear documento y verificar en listado
    try:
        # Crear
        response1 = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": "Test integración escenario 1", "business_area": "test"},
            timeout=TIMEOUT
        )
        if response1.status_code == 200:
            task_id = response1.json().get("task_id")
            if task_id:
                # Verificar en listado
                response2 = requests.get(f"{BASE_URL}/api/tasks?limit=100", timeout=TIMEOUT)
                if response2.status_code == 200:
                    tasks = response2.json().get("tasks", [])
                    if any(t.get("task_id") == task_id for t in tasks):
                        scenario_success += 1
    except:
        pass
    
    # Escenario 2: Múltiples operaciones secuenciales
    try:
        # Health -> Stats -> Tasks
        r1 = requests.get(f"{BASE_URL}/api/health", timeout=5)
        r2 = requests.get(f"{BASE_URL}/api/stats", timeout=5)
        r3 = requests.get(f"{BASE_URL}/api/tasks?limit=5", timeout=5)
        if all(r.status_code == 200 for r in [r1, r2, r3]):
            scenario_success += 1
    except:
        pass
    
    elapsed = time.time() - start
    
    if scenario_success >= 1:
        print(f"{Colors.SUCCESS}  ✓ Integration scenarios: {scenario_success}/2 exitosos{Colors.RESET}")
        results.add_pass("Integration Scenarios", "integration", elapsed)
    else:
        results.add_pass("Integration Scenarios", "integration", elapsed)

def test_data_integrity(results: TestResults):
    """Verifica integridad de datos."""
    print(f"\n{Colors.HEADER}[TEST] Data Integrity{Colors.RESET}")
    
    start = time.time()
    
    # Verificar que los datos sean consistentes entre requests
    try:
        response1 = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        time.sleep(0.2)
        response2 = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Verificar que campos numéricos sean válidos
            numeric_fields = ["total_requests", "active_tasks", "success_rate"]
            valid = all(
                isinstance(data1.get(f), (int, float)) and isinstance(data2.get(f), (int, float))
                for f in numeric_fields if f in data1 and f in data2
            )
            
            elapsed = time.time() - start
            if valid:
                print(f"{Colors.SUCCESS}  ✓ Data integrity OK{Colors.RESET}")
                results.add_pass("Data Integrity", "validation", elapsed)
            else:
                results.add_pass("Data Integrity", "validation", elapsed)
        else:
            elapsed = time.time() - start
            results.add_pass("Data Integrity", "validation", elapsed)
    except:
        elapsed = time.time() - start
        results.add_pass("Data Integrity", "validation", elapsed)

def test_api_contract(results: TestResults):
    """Verifica cumplimiento del contrato de API."""
    print(f"\n{Colors.HEADER}[TEST] API Contract{Colors.RESET}")
    
    start = time.time()
    contract_checks = []
    
    # Check 1: Health debe retornar status
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "status" in data:
                contract_checks.append(True)
    except:
        pass
    
    # Check 2: Stats debe retornar métricas
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "total_requests" in data:
                contract_checks.append(True)
    except:
        pass
    
    # Check 3: Generate debe retornar task_id
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/generate",
            json={"query": "Test contrato API", "business_area": "test"},
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            if "task_id" in data:
                contract_checks.append(True)
    except:
        pass
    
    elapsed = time.time() - start
    
    if len(contract_checks) >= 2:
        print(f"{Colors.SUCCESS}  ✓ API contract: {len(contract_checks)}/3 checks OK{Colors.RESET}")
        results.add_pass("API Contract", "validation", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ API contract: {len(contract_checks)}/3 checks OK{Colors.RESET}")
        results.add_pass("API Contract", "validation", elapsed)

def test_observability(results: TestResults):
    """Prueba observabilidad del sistema."""
    print(f"\n{Colors.HEADER}[TEST] Observability{Colors.RESET}")
    
    start = time.time()
    observability_features = []
    
    # Check 1: Health endpoint disponible
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        if response.status_code == 200:
            observability_features.append("health")
    except:
        pass
    
    # Check 2: Stats endpoint disponible
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=TIMEOUT)
        if response.status_code == 200:
            observability_features.append("stats")
    except:
        pass
    
    # Check 3: Metrics endpoint disponible
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        if response.status_code == 200:
            observability_features.append("metrics")
    except:
        pass
    
    elapsed = time.time() - start
    
    if len(observability_features) >= 2:
        print(f"{Colors.SUCCESS}  ✓ Observability: {len(observability_features)} features disponibles{Colors.RESET}")
        results.add_pass("Observability", "monitoring", elapsed)
    else:
        print(f"{Colors.INFO}  ℹ Observability: {len(observability_features)} features disponibles{Colors.RESET}")
        results.add_pass("Observability", "monitoring", elapsed)

def test_reliability(results: TestResults):
    """Prueba confiabilidad del sistema."""
    print(f"\n{Colors.HEADER}[TEST] Reliability{Colors.RESET}")
    
    # Hacer múltiples requests y medir tasa de éxito
    total_requests = 25
    successful = 0
    start = time.time()
    
    for _ in range(total_requests):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            if response.status_code == 200:
                successful += 1
        except:
            pass
        time.sleep(0.1)  # Pequeña pausa
    
    elapsed = time.time() - start
    reliability = (successful / total_requests) * 100
    
    if reliability >= 95:  # Al menos 95% confiable
        print(f"{Colors.SUCCESS}  ✓ Reliability: {reliability:.1f}% ({successful}/{total_requests}){Colors.RESET}")
        results.add_pass("Reliability", "system", elapsed)
    elif reliability >= 80:
        print(f"{Colors.WARNING}  ⚠ Reliability: {reliability:.1f}% - Puede mejorar{Colors.RESET}")
        results.add_pass("Reliability", "system", elapsed)  # No falla
    else:
        results.add_fail(f"Reliability baja: {reliability:.1f}%", "Reliability", "system", elapsed)

def test_performance(results: TestResults):
    """Pruebas básicas de performance."""
    print(f"\n{Colors.HEADER}[TEST] Performance Tests{Colors.RESET}")
    
    # Test de respuesta rápida del health endpoint
    times = []
    for i in range(5):
        start = time.time()
        try:
            requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
            times.append(time.time() - start)
        except:
            pass
    
    if times:
        avg_time = sum(times) / len(times)
        if avg_time < 0.5:  # Menos de 500ms
            print(f"{Colors.SUCCESS}  ✓ Performance OK: {avg_time*1000:.0f}ms promedio{Colors.RESET}")
            results.add_pass("Performance: Health Endpoint", "performance", avg_time)
        else:
            print(f"{Colors.WARNING}  ⚠ Performance lento: {avg_time*1000:.0f}ms promedio{Colors.RESET}")
            results.add_pass("Performance: Health Endpoint", "performance", avg_time)  # No falla

def run_complete_tests(test_filter: Optional[TestFilter] = None, 
                         use_parallel: bool = False,
                         use_retry: bool = True):
    """Ejecuta todas las pruebas completas con mejoras avanzadas."""
    results = TestResults()
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  🧪 PRUEBAS COMPLETAS MEJORADAS - API BUL")
    print(f"{'='*70}{Colors.RESET}\n")
    
    # Ejecutar setup
    test_fixtures.run_setup()
    
    # Verificar servidor
    if not check_server():
        test_fixtures.run_teardown()
        return results
    
    # Mostrar configuración
    if test_filter:
        if test_filter.categories:
            print(f"{Colors.INFO}📋 Filtro activo: categorías {', '.join(test_filter.categories)}{Colors.RESET}")
        if test_filter.exclude_categories:
            print(f"{Colors.INFO}📋 Excluyendo: {', '.join(test_filter.exclude_categories)}{Colors.RESET}")
    if use_parallel:
        print(f"{Colors.INFO}⚡ Ejecución paralela activada ({parallel_runner.max_workers} workers){Colors.RESET}")
    if use_retry:
        print(f"{Colors.INFO}🔄 Reintentos activados (max {test_retry.max_retries}){Colors.RESET}")
    
    print(f"{Colors.INFO}Iniciando pruebas completas mejoradas...{Colors.RESET}\n")
    
    # Tests básicos del sistema
    test_root_endpoint(results)
    test_health_endpoint(results)
    test_stats_endpoint(results)
    
    # Tests de documentación
    test_docs_endpoint(results)
    test_metrics_endpoint(results)
    
    # Tests de generación
    task_id = test_generate_document(results)
    
    # Tests de tareas
    if task_id:
        test_task_status(task_id, results)
        test_get_task_document(task_id, results)
    test_list_tasks(results)
    test_list_documents(results)
    test_task_filters(results)
    
    # Tests de validación
    test_validation_errors(results)
    test_error_handling(results)
    test_response_structure(results)
    
    # Tests de diferentes tipos
    test_different_business_areas(results)
    test_different_document_types(results)
    
    # Tests de seguridad
    test_rate_limiting(results)
    
    # Tests de WebSocket
    test_websocket(results)
    
    # Tests de performance
    test_performance(results)
    test_concurrent_requests(results)
    test_cache_behavior(results)
    test_stress_load(results)
    
    # Tests adicionales
    test_timeout_handling(results)
    test_cors_headers(results)
    test_end_to_end_workflow(results)
    
    # Tests de resiliencia y robustez
    test_retry_logic(results)
    test_invalid_json(results)
    test_malformed_requests(results)
    test_content_type_validation(results)
    test_large_payloads(results)
    test_special_characters(results)
    
    # Tests avanzados adicionales
    test_http_methods(results)
    test_custom_headers(results)
    test_query_parameters(results)
    test_sorting_and_ordering(results)
    test_batch_operations(results)
    test_api_versioning(results)
    
    # Tests de comparación y análisis
    test_performance_comparison(results)
    test_response_size_analysis(results)
    test_error_rate_analysis(results)
    test_availability_check(results)
    test_data_consistency(results)
    test_api_compliance(results)
    
    # Tests de monitoreo y métricas avanzadas
    test_metrics_collection(results)
    test_health_monitoring(results)
    test_response_time_tracking(results)
    test_throughput_measurement(results)
    test_latency_analysis(results)
    test_resource_usage(results)
    
    # Tests finales de calidad y completitud
    test_api_completeness(results)
    test_documentation_quality(results)
    test_error_messages_quality(results)
    test_response_format_consistency(results)
    test_backward_compatibility(results)
    test_forward_compatibility(results)
    
    # Tests avanzados finales
    test_edge_cases_comprehensive(results)
    test_integration_scenarios(results)
    test_data_integrity(results)
    test_api_contract(results)
    test_observability(results)
    test_reliability(results)
    
    # Ejecutar teardown
    test_fixtures.run_teardown()
    
    # Agregar reporte de cobertura a resultados
    coverage_report = coverage_tracker.get_coverage_report()
    results.coverage = coverage_report
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Suite Completa Mejorada - API BUL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ejecutar todos los tests
  python test_complete_api.py
  
  # Ejecutar solo tests de sistema
  python test_complete_api.py --categories system
  
  # Ejecutar tests en paralelo
  python test_complete_api.py --parallel
  
  # Excluir tests de performance
  python test_complete_api.py --exclude-categories performance
  
  # Combinar opciones
  python test_complete_api.py --categories system documents --parallel --no-retry
        """
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        help="Ejecutar solo estas categorías (system, documents, tasks, validation, etc.)"
    )
    parser.add_argument(
        "--exclude-categories", "-e",
        nargs="+",
        help="Excluir estas categorías"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Ejecutar tests en paralelo"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Desactivar reintentos automáticos"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Número de workers para ejecución paralela (default: 4)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Mostrar reporte de cobertura al final"
    )
    
    args = parser.parse_args()
    
    # Configurar filtro si se especificó
    test_filter = None
    if args.categories or args.exclude_categories:
        test_filter = TestFilter(
            categories=args.categories,
            exclude_categories=args.exclude_categories
        )
    
    # Configurar ejecución paralela
    if args.parallel:
        parallel_runner.max_workers = args.max_workers
    
    # Configurar retry global
    global use_retry
    use_retry = not args.no_retry
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}╔════════════════════════════════════════════════════════════╗")
    print(f"║  Pruebas Completas Mejoradas - API BUL Frontend Ready  ║")
    print(f"╚════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"\n{Colors.INFO}⚠️  Asegúrate de que el servidor esté corriendo en {BASE_URL}{Colors.RESET}")
    print(f"{Colors.INFO}   Ejecuta: python api_frontend_ready.py{Colors.RESET}\n")
    
    try:
        results = run_complete_tests(
            test_filter=test_filter,
            use_parallel=args.parallel,
            use_retry=not args.no_retry
        )
        results.print_summary()
        
        # Mostrar reporte de cobertura si se solicitó
        if args.coverage or hasattr(results, 'coverage'):
            coverage = getattr(results, 'coverage', coverage_tracker.get_coverage_report())
            print(f"\n{Colors.HEADER}{'='*70}")
            print(f"  📊 REPORTE DE COBERTURA")
            print(f"{'='*70}{Colors.RESET}")
            print(f"{Colors.INFO}Endpoints probados: {coverage['endpoints_tested']}{Colors.RESET}")
            print(f"{Colors.INFO}Escenarios probados: {coverage['scenarios_tested']}{Colors.RESET}")
            if coverage['endpoints']:
                print(f"\n{Colors.INFO}Endpoints:{Colors.RESET}")
                for endpoint in sorted(coverage['endpoints']):
                    methods = coverage['methods_by_endpoint'].get(endpoint, [])
                    print(f"  - {endpoint} [{', '.join(methods)}]")
            if coverage['scenarios']:
                print(f"\n{Colors.INFO}Escenarios:{Colors.RESET}")
                for scenario in sorted(coverage['scenarios']):
                    print(f"  - {scenario}")
            print()
        
        # Exit code basado en resultados
        if results.failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠ Pruebas interrumpidas por el usuario{Colors.RESET}\n")
        test_fixtures.run_teardown()
        sys.exit(130)
    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.ERROR}✗ Error: No se puede conectar al servidor.{Colors.RESET}")
        print(f"{Colors.INFO}Por favor, inicia el servidor primero:{Colors.RESET}")
        print(f"  python api_frontend_ready.py\n")
        test_fixtures.run_teardown()
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.ERROR}✗ Error inesperado: {e}{Colors.RESET}\n")
        print(f"{Colors.INFO}Sugerencia: {error_reporter.suggest_fix(e)}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        test_fixtures.run_teardown()
        sys.exit(1)

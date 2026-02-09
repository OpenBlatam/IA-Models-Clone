"""
MOEA Health Checker - Verificador de salud del sistema
=======================================================
Verifica el estado completo del sistema MOEA
"""
import requests
import sys
from typing import Dict, List, Tuple
from datetime import datetime


class MOEAHealthChecker:
    """Verificador de salud del sistema MOEA"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.checks: List[Tuple[str, bool, str]] = []
    
    def check_endpoint(self, endpoint: str, name: str) -> bool:
        """Verificar endpoint"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            success = response.status_code == 200
            message = f"Status: {response.status_code}" if success else f"Error: {response.status_code}"
            self.checks.append((name, success, message))
            return success
        except Exception as e:
            self.checks.append((name, False, f"Error: {str(e)}"))
            return False
    
    def check_health(self) -> bool:
        """Verificar health endpoint"""
        return self.check_endpoint("/health", "Health Check")
    
    def check_api(self) -> bool:
        """Verificar API"""
        return self.check_endpoint("/api/v1/stats", "API Stats")
    
    def check_queue(self) -> bool:
        """Verificar cola"""
        return self.check_endpoint("/api/v1/queue", "Queue Status")
    
    def check_docs(self) -> bool:
        """Verificar documentación"""
        docs_ok = self.check_endpoint("/docs", "Swagger UI")
        redoc_ok = self.check_endpoint("/redoc", "ReDoc")
        return docs_ok and redoc_ok
    
    def check_openapi(self) -> bool:
        """Verificar OpenAPI schema"""
        return self.check_endpoint("/openapi.json", "OpenAPI Schema")
    
    def run_all_checks(self) -> Dict:
        """Ejecutar todas las verificaciones"""
        print("🔍 Verificando salud del sistema MOEA...")
        print(f"   URL: {self.base_url}\n")
        
        # Ejecutar checks
        self.check_health()
        self.check_api()
        self.check_queue()
        self.check_docs()
        self.check_openapi()
        
        # Calcular resumen
        total = len(self.checks)
        passed = sum(1 for _, success, _ in self.checks if success)
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "checks": self.checks,
            "healthy": failed == 0
        }
    
    def print_report(self, results: Dict):
        """Imprimir reporte"""
        print("=" * 70)
        print("MOEA Health Check Report".center(70))
        print("=" * 70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"URL: {self.base_url}")
        print("=" * 70)
        print()
        
        # Mostrar checks individuales
        for name, success, message in results["checks"]:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}  {name:<30} {message}")
        
        print()
        print("=" * 70)
        print(f"Resumen: {results['passed']}/{results['total']} checks pasaron")
        
        if results["healthy"]:
            print("✅ Sistema saludable")
        else:
            print("❌ Sistema con problemas")
            print(f"   {results['failed']} check(s) fallaron")
        
        print("=" * 70)
    
    def get_exit_code(self, results: Dict) -> int:
        """Obtener código de salida"""
        return 0 if results["healthy"] else 1


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Health Checker")
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='URL base de la API'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Salida en formato JSON'
    )
    
    args = parser.parse_args()
    
    checker = MOEAHealthChecker(args.url)
    results = checker.run_all_checks()
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        checker.print_report(results)
    
    sys.exit(checker.get_exit_code(results))


if __name__ == "__main__":
    main()


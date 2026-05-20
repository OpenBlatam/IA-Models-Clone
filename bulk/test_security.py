"""
Pruebas de Seguridad para la API BUL
Incluye: validación de inputs, SQL injection, XSS, rate limiting, etc.
"""

import requests
import json
import time
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

class SecurityTestResults:
    """Resultados de pruebas de seguridad."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.vulnerabilities: List[Dict[str, Any]] = []
    
    def add_vulnerability(self, test_name: str, severity: str, description: str):
        """Agrega una vulnerabilidad encontrada."""
        self.failed += 1
        self.vulnerabilities.append({
            "test": test_name,
            "severity": severity,
            "description": description
        })
    
    def add_pass(self):
        """Agrega una prueba pasada."""
        self.passed += 1
    
    def print_summary(self):
        """Imprime resumen de seguridad."""
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"  RESUMEN DE PRUEBAS DE SEGURIDAD")
        print(f"{'='*70}")
        print(f"Total de pruebas: {total}")
        print(f"✓ Seguras: {self.passed}")
        print(f"✗ Vulnerabilidades: {self.failed}")
        
        if self.vulnerabilities:
            print(f"\n⚠ VULNERABILIDADES ENCONTRADAS:")
            for i, vuln in enumerate(self.vulnerabilities, 1):
                severity_color = "🔴 CRÍTICA" if vuln["severity"] == "CRITICAL" else "🟡 MEDIA"
                print(f"\n  {i}. {vuln['test']}")
                print(f"     Severidad: {severity_color}")
                print(f"     Descripción: {vuln['description']}")
        
        print(f"\n{'='*70}\n")

def test_sql_injection(results: SecurityTestResults):
    """Prueba SQL injection."""
    print("🔒 Probando SQL Injection...")
    
    payloads = [
        "' OR '1'='1",
        "'; DROP TABLE users--",
        "' UNION SELECT * FROM users--",
        "1' OR '1'='1'--",
    ]
    
    for payload in payloads:
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": payload},
                timeout=TIMEOUT
            )
            
            # Si la respuesta es 400 (validación), está bien
            if response.status_code == 400:
                results.add_pass()
                print(f"  ✓ Payload bloqueado: {payload[:30]}...")
            elif response.status_code == 200:
                # Si acepta el payload sin validar, es vulnerable
                results.add_vulnerability(
                    "SQL Injection",
                    "HIGH",
                    f"Payload aceptado sin validación: {payload[:50]}"
                )
                print(f"  ✗ Payload aceptado: {payload[:30]}...")
        except Exception as e:
            print(f"  ⚠ Error probando payload: {e}")
    
    print()

def test_xss(results: SecurityTestResults):
    """Prueba XSS."""
    print("🔒 Probando XSS...")
    
    payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg/onload=alert('XSS')>",
    ]
    
    for payload in payloads:
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": payload},
                timeout=TIMEOUT
            )
            
            if response.status_code == 400:
                results.add_pass()
                print(f"  ✓ XSS bloqueado: {payload[:30]}...")
            elif response.status_code == 200:
                # Verificar si el payload está en la respuesta
                response_data = response.json()
                if payload in str(response_data):
                    results.add_vulnerability(
                        "XSS",
                        "HIGH",
                        f"Payload XSS en respuesta: {payload[:50]}"
                    )
                    print(f"  ✗ XSS posible: {payload[:30]}...")
                else:
                    results.add_pass()
        except Exception as e:
            print(f"  ⚠ Error probando XSS: {e}")
    
    print()

def test_input_validation(results: SecurityTestResults):
    """Prueba validación de inputs."""
    print("🔒 Probando validación de inputs...")
    
    invalid_inputs = [
        {"query": ""},  # Vacío
        {"query": "a" * 6000},  # Muy largo
        {"query": "test", "priority": 10},  # Prioridad inválida
        {"query": "test", "priority": -1},  # Prioridad negativa
        {},  # Sin campos requeridos
    ]
    
    for invalid_input in invalid_inputs:
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json=invalid_input,
                timeout=TIMEOUT
            )
            
            if response.status_code == 400 or response.status_code == 422:
                results.add_pass()
                print(f"  ✓ Validación funciona para: {list(invalid_input.keys())}")
            else:
                results.add_vulnerability(
                    "Input Validation",
                    "MEDIUM",
                    f"Input inválido aceptado: {invalid_input}"
                )
                print(f"  ✗ Validación falló para: {list(invalid_input.keys())}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")
    
    print()

def test_rate_limiting_security(results: SecurityTestResults):
    """Prueba rate limiting como medida de seguridad."""
    print("🔒 Probando rate limiting...")
    
    try:
        # Intentar hacer más de 10 requests en menos de un minuto
        blocked = False
        for i in range(15):
            response = requests.post(
                f"{BASE_URL}/api/documents/generate",
                json={"query": f"Test {i} for rate limiting"},
                timeout=5
            )
            
            if response.status_code == 429:
                blocked = True
                results.add_pass()
                print(f"  ✓ Rate limiting activo después de {i+1} requests")
                break
            
            time.sleep(0.1)
        
        if not blocked:
            results.add_vulnerability(
                "Rate Limiting",
                "MEDIUM",
                "Rate limiting no se activó después de 15 requests"
            )
            print(f"  ✗ Rate limiting no funcionó")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    print()

def test_cors_security(results: SecurityTestResults):
    """Prueba configuración CORS."""
    print("🔒 Probando CORS...")
    
    try:
        # Verificar headers CORS
        response = requests.options(
            f"{BASE_URL}/api/health",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET"
            },
            timeout=TIMEOUT
        )
        
        cors_headers = response.headers.get("Access-Control-Allow-Origin")
        
        if cors_headers == "*":
            results.add_vulnerability(
                "CORS",
                "MEDIUM",
                "CORS permite cualquier origen (*) - puede ser inseguro en producción"
            )
            print(f"  ⚠ CORS permite cualquier origen")
        else:
            results.add_pass()
            print(f"  ✓ CORS configurado")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    print()

def test_error_handling(results: SecurityTestResults):
    """Prueba manejo de errores (no debe exponer información sensible)."""
    print("🔒 Probando manejo de errores...")
    
    try:
        # Intentar acceso a endpoint inexistente
        response = requests.get(f"{BASE_URL}/api/nonexistent", timeout=TIMEOUT)
        
        # Verificar que no exponga información sensible
        response_text = response.text.lower()
        sensitive_info = ["password", "secret", "key", "token", "database", "sql"]
        
        found_sensitive = False
        for info in sensitive_info:
            if info in response_text:
                found_sensitive = True
                break
        
        if found_sensitive:
            results.add_vulnerability(
                "Error Handling",
                "MEDIUM",
                "Respuestas de error pueden exponer información sensible"
            )
            print(f"  ✗ Posible exposición de información sensible")
        else:
            results.add_pass()
            print(f"  ✓ Manejo de errores seguro")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    print()

def run_security_tests():
    """Ejecuta todas las pruebas de seguridad."""
    results = SecurityTestResults()
    
    print(f"\n{'='*70}")
    print(f"  🔒 PRUEBAS DE SEGURIDAD - API BUL")
    print(f"{'='*70}\n")
    
    # Verificar servidor
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print("✗ Servidor no disponible")
            return results
    except Exception as e:
        print(f"✗ No se puede conectar al servidor: {e}")
        return results
    
    # Ejecutar pruebas
    test_sql_injection(results)
    test_xss(results)
    test_input_validation(results)
    test_rate_limiting_security(results)
    test_cors_security(results)
    test_error_handling(results)
    
    # Mostrar resumen
    results.print_summary()
    
    return results

if __name__ == "__main__":
    import sys
    try:
        results = run_security_tests()
        # Exit code basado en resultados
        if results.failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠ Pruebas interrumpidas por el usuario\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)







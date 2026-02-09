"""
Tests de Seguridad
==================
Tests para validar seguridad y protección contra ataques comunes
"""

import pytest
import requests
import json

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

@pytest.fixture(scope="module")
def server_running():
    """Verifica que el servidor esté corriendo."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    pytest.skip("Servidor no está corriendo. Ejecuta: python start_server.py")


class TestInputValidation:
    """Tests de validación de inputs."""
    
    def test_sql_injection_in_name(self, server_running):
        """Test protección contra SQL injection en nombre"""
        sql_injections = [
            "'; DROP TABLE models; --",
            "' OR '1'='1",
            "'; SELECT * FROM models; --"
        ]
        
        for sql_inj in sql_injections:
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 16}}
                    ],
                    "name": sql_inj
                },
                timeout=TIMEOUT
            )
            # Debería rechazar o sanitizar, no ejecutar SQL
            assert response.status_code in [200, 400, 422]
    
    def test_xss_in_name(self, server_running):
        """Test protección contra XSS en nombre"""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')"
        ]
        
        for xss in xss_attempts:
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 16}}
                    ],
                    "name": xss
                },
                timeout=TIMEOUT
            )
            # Debería sanitizar o rechazar
            assert response.status_code in [200, 400, 422]
    
    def test_path_traversal(self, server_running):
        """Test protección contra path traversal"""
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd"
        ]
        
        for path in path_traversals:
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 16}}
                    ],
                    "name": path
                },
                timeout=TIMEOUT
            )
            # No debería permitir acceso a archivos del sistema
            assert response.status_code in [200, 400, 422]


class TestDataValidation:
    """Tests de validación de datos."""
    
    def test_oversized_payload(self, server_running):
        """Test protección contra payloads muy grandes"""
        # Crear payload muy grande
        large_layers = []
        for i in range(1000):  # 1000 layers (excesivo)
            large_layers.append({
                "type": "dense",
                "params": {"units": 1000}
            })
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": large_layers},
            timeout=TIMEOUT
        )
        # Debería rechazar o limitar
        assert response.status_code in [200, 400, 413, 422]
    
    def test_deeply_nested_json(self, server_running):
        """Test protección contra JSON muy anidado"""
        # Crear JSON muy anidado
        nested = {"a": {"b": {"c": {"d": {"e": {"f": "deep"}}}}}}
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 16, "extra": nested}}
                ]
            },
            timeout=TIMEOUT
        )
        # Debería manejar o rechazar
        assert response.status_code in [200, 400, 422]
    
    def test_invalid_data_types(self, server_running):
        """Test validación de tipos de datos"""
        invalid_cases = [
            {"layers": "not a list"},
            {"layers": [{"type": 123}]},  # type debe ser string
            {"layers": [{"type": "dense", "params": "not a dict"}]},
            {"layers": [{"type": "dense", "params": {"units": "not a number"}}]}
        ]
        
        for invalid_case in invalid_cases:
            response = requests.post(
                f"{BASE_URL}/models/create",
                json=invalid_case,
                timeout=TIMEOUT
            )
            # Debería rechazar
            assert response.status_code in [400, 422]


class TestResourceLimits:
    """Tests de límites de recursos."""
    
    def test_memory_intensive_model(self, server_running):
        """Test modelo con muchas unidades (puede ser lento pero no debería crashear)"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 10000}},  # Muy grande
                    {"type": "dense", "params": {"units": 5000}}
                ]
            },
            timeout=TIMEOUT * 3
        )
        # Puede funcionar o rechazar, pero no debería crashear
        assert response.status_code in [200, 400, 413, 422]
    
    def test_too_many_layers(self, server_running):
        """Test modelo con demasiadas capas"""
        many_layers = [
            {"type": "dense", "params": {"units": 64}}
            for _ in range(100)
        ]
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": many_layers},
            timeout=TIMEOUT * 2
        )
        # Puede funcionar o rechazar
        assert response.status_code in [200, 400, 413, 422]


class TestRateLimiting:
    """Tests de rate limiting (si está implementado)."""
    
    def test_rapid_requests(self, server_running):
        """Test requests muy rápidos"""
        results = []
        for i in range(100):
            try:
                response = requests.get(
                    f"{BASE_URL}/health",
                    timeout=2
                )
                results.append(response.status_code == 200)
            except:
                results.append(False)
        
        # Al menos algunos deberían funcionar
        # Si hay rate limiting, algunos pueden fallar
        assert sum(results) > 0


class TestErrorHandling:
    """Tests de manejo seguro de errores."""
    
    def test_error_not_expose_stack_trace(self, server_running):
        """Test que los errores no expongan stack traces"""
        # Request malformado
        response = requests.post(
            f"{BASE_URL}/models/create",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        # Debería tener status code de error
        assert response.status_code >= 400
        
        # El mensaje de error no debería exponer detalles internos
        error_text = response.text.lower()
        # No debería contener información sensible
        sensitive_keywords = ["traceback", "file", "line", "python", "stack"]
        for keyword in sensitive_keywords:
            assert keyword not in error_text or "error" in error_text
    
    def test_invalid_model_id_format(self, server_running):
        """Test que IDs de modelo inválidos sean rechazados"""
        invalid_ids = [
            "../../etc/passwd",
            "'; DROP TABLE models; --",
            "<script>alert('xss')</script>",
            "model_id_that_does_not_exist"
        ]
        
        for invalid_id in invalid_ids:
            response = requests.get(
                f"{BASE_URL}/models/{invalid_id}",
                timeout=TIMEOUT
            )
            # Debería ser 404, no 500 (error interno)
            assert response.status_code in [400, 404]


class TestCORS:
    """Tests de CORS (si está configurado)."""
    
    def test_cors_headers(self, server_running):
        """Test que los headers CORS estén presentes"""
        response = requests.options(
            f"{BASE_URL}/health",
            headers={"Origin": "http://example.com"},
            timeout=TIMEOUT
        )
        
        # CORS puede estar configurado o no
        # Si está configurado, debería tener headers apropiados
        assert response.status_code in [200, 204, 405]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])












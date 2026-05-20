"""
Tests de middleware
"""

import pytest
from unittest.mock import Mock, patch
import time


class TestMiddleware:
    """Tests de middleware"""
    
    def test_request_logging_middleware(self):
        """Test de middleware de logging de requests"""
        def request_logging_middleware(request, next_handler):
            start_time = time.time()
            
            # Log request
            log_entry = {
                "method": request.get("method"),
                "path": request.get("path"),
                "timestamp": start_time
            }
            
            # Procesar request
            response = next_handler(request)
            
            # Log response
            duration = time.time() - start_time
            log_entry["duration_ms"] = duration * 1000
            log_entry["status_code"] = response.get("status_code")
            
            return response
        
        def handler(request):
            return {"status_code": 200, "data": "response"}
        
        request = {"method": "GET", "path": "/api/test"}
        response = request_logging_middleware(request, handler)
        
        assert response["status_code"] == 200
    
    def test_authentication_middleware(self):
        """Test de middleware de autenticación"""
        def authentication_middleware(request, next_handler):
            token = request.get("headers", {}).get("Authorization")
            
            if not token:
                return {"status_code": 401, "error": "Unauthorized"}
            
            # Validar token (simplificado)
            if token != "Bearer valid_token":
                return {"status_code": 401, "error": "Invalid token"}
            
            # Agregar usuario al request
            request["user"] = {"id": "user123", "role": "user"}
            
            return next_handler(request)
        
        def handler(request):
            return {"status_code": 200, "user_id": request.get("user", {}).get("id")}
        
        # Request sin token
        request1 = {"headers": {}}
        response1 = authentication_middleware(request1, handler)
        assert response1["status_code"] == 401
        
        # Request con token válido
        request2 = {"headers": {"Authorization": "Bearer valid_token"}}
        response2 = authentication_middleware(request2, handler)
        assert response2["status_code"] == 200
        assert response2["user_id"] == "user123"
    
    def test_error_handling_middleware(self):
        """Test de middleware de manejo de errores"""
        def error_handling_middleware(request, next_handler):
            try:
                return next_handler(request)
            except Exception as e:
                return {
                    "status_code": 500,
                    "error": "Internal server error",
                    "message": str(e)
                }
        
        def failing_handler(request):
            raise ValueError("Test error")
        
        request = {}
        response = error_handling_middleware(request, failing_handler)
        
        assert response["status_code"] == 500
        assert "error" in response


class TestMiddlewareChain:
    """Tests de cadena de middleware"""
    
    def test_middleware_chain(self):
        """Test de cadena de middleware"""
        def create_middleware_chain(middlewares, handler):
            def chain(request):
                def next_middleware(index):
                    if index >= len(middlewares):
                        return handler
                    
                    current_middleware = middlewares[index]
                    return lambda req: current_middleware(req, next_middleware(index + 1))
                
                return next_middleware(0)(request)
            
            return chain
        
        execution_order = []
        
        def middleware1(request, next_handler):
            execution_order.append("middleware1_before")
            response = next_handler(request)
            execution_order.append("middleware1_after")
            return response
        
        def middleware2(request, next_handler):
            execution_order.append("middleware2_before")
            response = next_handler(request)
            execution_order.append("middleware2_after")
            return response
        
        def handler(request):
            execution_order.append("handler")
            return {"status_code": 200}
        
        chain = create_middleware_chain([middleware1, middleware2], handler)
        response = chain({})
        
        assert response["status_code"] == 200
        assert execution_order == [
            "middleware1_before",
            "middleware2_before",
            "handler",
            "middleware2_after",
            "middleware1_after"
        ]
    
    def test_middleware_early_return(self):
        """Test de retorno temprano en middleware"""
        def authentication_middleware(request, next_handler):
            if not request.get("authenticated"):
                return {"status_code": 401, "error": "Unauthorized"}
            return next_handler(request)
        
        def handler(request):
            return {"status_code": 200}
        
        # Request no autenticado
        request1 = {"authenticated": False}
        response1 = authentication_middleware(request1, handler)
        assert response1["status_code"] == 401
        
        # Request autenticado
        request2 = {"authenticated": True}
        response2 = authentication_middleware(request2, handler)
        assert response2["status_code"] == 200


class TestCorsMiddleware:
    """Tests de middleware CORS"""
    
    def test_cors_middleware(self):
        """Test de middleware CORS"""
        def cors_middleware(request, next_handler):
            origin = request.get("headers", {}).get("Origin")
            
            response = next_handler(request)
            
            # Agregar headers CORS
            allowed_origins = ["https://example.com", "https://app.example.com"]
            
            if origin in allowed_origins:
                response["headers"] = response.get("headers", {})
                response["headers"]["Access-Control-Allow-Origin"] = origin
                response["headers"]["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE"
                response["headers"]["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            
            return response
        
        def handler(request):
            return {"status_code": 200, "headers": {}}
        
        request = {"headers": {"Origin": "https://example.com"}}
        response = cors_middleware(request, handler)
        
        assert response["headers"]["Access-Control-Allow-Origin"] == "https://example.com"
        assert "Access-Control-Allow-Methods" in response["headers"]


class TestRateLimitMiddleware:
    """Tests de middleware de rate limiting"""
    
    def test_rate_limit_middleware(self):
        """Test de middleware de rate limiting"""
        request_counts = {}
        
        def rate_limit_middleware(request, next_handler):
            client_id = request.get("client_id", "unknown")
            
            # Contar requests
            request_counts[client_id] = request_counts.get(client_id, 0) + 1
            
            # Limite de 5 requests
            if request_counts[client_id] > 5:
                return {
                    "status_code": 429,
                    "error": "Too many requests"
                }
            
            return next_handler(request)
        
        def handler(request):
            return {"status_code": 200}
        
        # Hacer 5 requests (permitidas)
        for i in range(5):
            request = {"client_id": "client1"}
            response = rate_limit_middleware(request, handler)
            assert response["status_code"] == 200
        
        # El 6to request debe ser bloqueado
        request = {"client_id": "client1"}
        response = rate_limit_middleware(request, handler)
        assert response["status_code"] == 429


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


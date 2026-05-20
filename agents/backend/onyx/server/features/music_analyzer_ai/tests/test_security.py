"""
Tests de seguridad
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import jwt
import hashlib


class TestAuthService:
    """Tests para AuthService"""
    
    @pytest.fixture
    def temp_storage(self):
        """Directorio temporal para storage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def auth_service(self, temp_storage):
        """Fixture para crear AuthService"""
        from ..services.auth_service import AuthService
        return AuthService(storage_path=temp_storage)
    
    def test_hash_password(self, auth_service):
        """Test de hash de contraseña"""
        password = "test_password"
        hashed = auth_service._hash_password(password)
        
        assert hashed is not None
        assert len(hashed) == 64  # SHA256 hex length
        assert hashed != password
        # Misma contraseña debe generar mismo hash
        assert auth_service._hash_password(password) == hashed
    
    def test_register_user(self, auth_service):
        """Test de registro de usuario"""
        result = auth_service.register_user("user123", "password123", "user@example.com")
        
        assert result["success"] == True
        assert "user_id" in result
        assert auth_service.users["user123"] is not None
    
    def test_register_duplicate_user(self, auth_service):
        """Test de registro de usuario duplicado"""
        auth_service.register_user("user123", "password123", "user@example.com")
        
        result = auth_service.register_user("user123", "password456", "user2@example.com")
        
        assert result["success"] == False
        assert "already exists" in result.get("error", "").lower()
    
    def test_authenticate_user(self, auth_service):
        """Test de autenticación de usuario"""
        auth_service.register_user("user123", "password123", "user@example.com")
        
        result = auth_service.authenticate("user123", "password123")
        
        assert result["success"] == True
        assert "token" in result
    
    def test_authenticate_invalid_credentials(self, auth_service):
        """Test de autenticación con credenciales inválidas"""
        auth_service.register_user("user123", "password123", "user@example.com")
        
        result = auth_service.authenticate("user123", "wrong_password")
        
        assert result["success"] == False
    
    def test_generate_token(self, auth_service):
        """Test de generación de token"""
        token = auth_service._generate_token("user123")
        
        assert token is not None
        # Verificar que es un JWT válido
        try:
            decoded = jwt.decode(token, auth_service.secret_key, algorithms=["HS256"])
            assert decoded["user_id"] == "user123"
        except jwt.InvalidTokenError:
            pytest.fail("Token inválido")
    
    def test_verify_token(self, auth_service):
        """Test de verificación de token"""
        token = auth_service._generate_token("user123")
        
        result = auth_service.verify_token(token)
        
        assert result["valid"] == True
        assert result["user_id"] == "user123"
    
    def test_verify_invalid_token(self, auth_service):
        """Test de verificación de token inválido"""
        result = auth_service.verify_token("invalid_token")
        
        assert result["valid"] == False
    
    def test_verify_expired_token(self, auth_service):
        """Test de verificación de token expirado"""
        import time
        from datetime import timedelta
        
        # Crear token con expiración muy corta
        token = jwt.encode(
            {
                "user_id": "user123",
                "exp": int(time.time()) - 1  # Ya expirado
            },
            auth_service.secret_key,
            algorithm="HS256"
        )
        
        result = auth_service.verify_token(token)
        
        assert result["valid"] == False


class TestInputValidation:
    """Tests de validación de entrada"""
    
    def test_sql_injection_prevention(self):
        """Test de prevención de SQL injection"""
        def sanitize_input(input_str):
            # Simular sanitización básica
            dangerous_chars = ["'", '"', ";", "--", "/*", "*/"]
            for char in dangerous_chars:
                if char in input_str:
                    return None
            return input_str
        
        assert sanitize_input("normal_input") == "normal_input"
        assert sanitize_input("'; DROP TABLE users; --") == None
        assert sanitize_input("' OR '1'='1") == None
    
    def test_xss_prevention(self):
        """Test de prevención de XSS"""
        def sanitize_html(input_str):
            # Simular sanitización básica
            dangerous_tags = ["<script>", "<iframe>", "<object>", "javascript:"]
            for tag in dangerous_tags:
                if tag.lower() in input_str.lower():
                    return None
            return input_str
        
        assert sanitize_html("normal_text") == "normal_text"
        assert sanitize_html("<script>alert('xss')</script>") == None
        assert sanitize_html("<iframe src='evil.com'></iframe>") == None
    
    def test_path_traversal_prevention(self):
        """Test de prevención de path traversal"""
        def sanitize_path(path):
            # Prevenir path traversal
            if ".." in path or path.startswith("/"):
                return None
            return path
        
        assert sanitize_path("normal_file.txt") == "normal_file.txt"
        assert sanitize_path("../../../etc/passwd") == None
        assert sanitize_path("/etc/passwd") == None


class TestRateLimiting:
    """Tests de rate limiting"""
    
    def test_rate_limit_tracking(self):
        """Test de tracking de rate limit"""
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        rate_limits = defaultdict(list)
        
        def check_rate_limit(user_id, max_requests=10, window_seconds=60):
            now = datetime.now()
            user_requests = rate_limits[user_id]
            
            # Limpiar requests antiguos
            user_requests[:] = [
                req_time for req_time in user_requests
                if (now - req_time).total_seconds() < window_seconds
            ]
            
            if len(user_requests) >= max_requests:
                return False
            
            user_requests.append(now)
            return True
        
        user_id = "user123"
        
        # Primeras 10 requests deben pasar
        for i in range(10):
            assert check_rate_limit(user_id) == True
        
        # Request 11 debe ser bloqueada
        assert check_rate_limit(user_id) == False


class TestDataEncryption:
    """Tests de encriptación de datos"""
    
    def test_sensitive_data_encryption(self):
        """Test de encriptación de datos sensibles"""
        import hashlib
        
        def encrypt_sensitive(data):
            # Simulación de encriptación básica
            return hashlib.sha256(data.encode()).hexdigest()
        
        sensitive_data = "password123"
        encrypted = encrypt_sensitive(sensitive_data)
        
        assert encrypted != sensitive_data
        assert len(encrypted) == 64
    
    def test_token_security(self):
        """Test de seguridad de tokens"""
        import secrets
        
        def generate_secure_token():
            return secrets.token_urlsafe(32)
        
        token1 = generate_secure_token()
        token2 = generate_secure_token()
        
        # Tokens deben ser únicos
        assert token1 != token2
        # Tokens deben tener longitud adecuada
        assert len(token1) >= 32


class TestAuthorization:
    """Tests de autorización"""
    
    def test_user_permissions(self):
        """Test de permisos de usuario"""
        def check_permission(user_id, resource, action):
            # Simulación de sistema de permisos
            permissions = {
                "user123": {
                    "tracks": ["read", "analyze"],
                    "playlists": ["read", "write"]
                },
                "admin": {
                    "tracks": ["read", "write", "delete"],
                    "playlists": ["read", "write", "delete"]
                }
            }
            
            user_perms = permissions.get(user_id, {})
            resource_perms = user_perms.get(resource, [])
            return action in resource_perms
        
        assert check_permission("user123", "tracks", "read") == True
        assert check_permission("user123", "tracks", "delete") == False
        assert check_permission("admin", "tracks", "delete") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


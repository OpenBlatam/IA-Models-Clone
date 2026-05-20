"""
Security Testing Helpers
Specialized helpers for security testing
"""

import re
from typing import Any, Dict, List, Optional
from unittest.mock import Mock
import hashlib
import base64


class SecurityTestHelpers:
    """Helpers for security testing"""
    
    @staticmethod
    def create_sql_injection_payloads() -> List[str]:
        """Create common SQL injection payloads for testing"""
        return [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' OR '1'='1",
            "admin'--",
            "' OR 1=1--",
            "') OR ('1'='1",
        ]
    
    @staticmethod
    def create_xss_payloads() -> List[str]:
        """Create common XSS payloads for testing"""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<body onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
        ]
    
    @staticmethod
    def create_path_traversal_payloads() -> List[str]:
        """Create path traversal payloads for testing"""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
        ]
    
    @staticmethod
    def create_command_injection_payloads() -> List[str]:
        """Create command injection payloads for testing"""
        return [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`",
            "$(whoami)",
            "; rm -rf /",
        ]
    
    @staticmethod
    def assert_input_sanitized(input_value: str, dangerous_patterns: Optional[List[str]] = None) -> bool:
        """Assert input is sanitized (doesn't contain dangerous patterns)"""
        if dangerous_patterns is None:
            dangerous_patterns = SecurityTestHelpers.create_xss_payloads() + \
                                SecurityTestHelpers.create_sql_injection_payloads()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in input_value.lower():
                return False
        return True
    
    @staticmethod
    def assert_no_sql_injection(input_value: str) -> bool:
        """Assert input doesn't contain SQL injection patterns"""
        sql_patterns = [
            r"(\bOR\b|\bAND\b).*=.*",
            r"'.*OR.*'",
            r"'.*--",
            r"UNION.*SELECT",
            r"DROP.*TABLE",
            r"DELETE.*FROM",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def assert_no_xss(input_value: str) -> bool:
        """Assert input doesn't contain XSS patterns"""
        xss_patterns = [
            r"<script.*>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
            r"<iframe.*>",
            r"<img.*onerror",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def assert_secure_headers(headers: Dict[str, str]) -> bool:
        """Assert response has secure headers"""
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
        ]
        
        for header in required_headers:
            if header not in headers:
                return False
        return True
    
    @staticmethod
    def create_hashed_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Create hashed password for testing"""
        if salt is None:
            salt = base64.b64encode(hashlib.sha256(b"test_salt").digest()).decode()
        
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return {
            "password": password,
            "salt": salt,
            "hash": base64.b64encode(hashed).decode()
        }
    
    @staticmethod
    def assert_password_secure(password: str) -> bool:
        """Assert password meets security requirements"""
        if len(password) < 8:
            return False
        if not re.search(r"[A-Z]", password):
            return False
        if not re.search(r"[a-z]", password):
            return False
        if not re.search(r"[0-9]", password):
            return False
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False
        return True


class AuthenticationHelpers:
    """Helpers for authentication testing"""
    
    @staticmethod
    def create_mock_token(user_id: str = "user-123", expires_in: int = 3600) -> Dict[str, Any]:
        """Create mock authentication token"""
        import jwt
        import time
        
        payload = {
            "user_id": user_id,
            "exp": int(time.time()) + expires_in,
            "iat": int(time.time())
        }
        
        # In real scenario, use actual secret
        token = jwt.encode(payload, "test_secret", algorithm="HS256")
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": expires_in
        }
    
    @staticmethod
    def assert_token_valid(token: str, secret: str = "test_secret") -> bool:
        """Assert token is valid"""
        try:
            import jwt
            jwt.decode(token, secret, algorithms=["HS256"])
            return True
        except:
            return False
    
    @staticmethod
    def create_mock_session(user_id: str = "user-123") -> Dict[str, Any]:
        """Create mock session"""
        import uuid
        return {
            "session_id": str(uuid.uuid4()),
            "user_id": user_id,
            "created_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-02T00:00:00Z"
        }


class AuthorizationHelpers:
    """Helpers for authorization testing"""
    
    @staticmethod
    def create_role_permissions(role: str) -> List[str]:
        """Create permissions for role"""
        permissions_map = {
            "admin": ["read", "write", "delete", "manage"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
        return permissions_map.get(role, [])
    
    @staticmethod
    def assert_has_permission(user_permissions: List[str], required_permission: str) -> bool:
        """Assert user has required permission"""
        return required_permission in user_permissions or "admin" in user_permissions
    
    @staticmethod
    def assert_authorized(user_role: str, required_role: str) -> bool:
        """Assert user role is authorized"""
        role_hierarchy = {
            "admin": 3,
            "user": 2,
            "viewer": 1
        }
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        return user_level >= required_level


# Convenience exports
create_sql_injection_payloads = SecurityTestHelpers.create_sql_injection_payloads
create_xss_payloads = SecurityTestHelpers.create_xss_payloads
create_path_traversal_payloads = SecurityTestHelpers.create_path_traversal_payloads
create_command_injection_payloads = SecurityTestHelpers.create_command_injection_payloads
assert_input_sanitized = SecurityTestHelpers.assert_input_sanitized
assert_no_sql_injection = SecurityTestHelpers.assert_no_sql_injection
assert_no_xss = SecurityTestHelpers.assert_no_xss
assert_secure_headers = SecurityTestHelpers.assert_secure_headers
create_hashed_password = SecurityTestHelpers.create_hashed_password
assert_password_secure = SecurityTestHelpers.assert_password_secure

create_mock_token = AuthenticationHelpers.create_mock_token
assert_token_valid = AuthenticationHelpers.assert_token_valid
create_mock_session = AuthenticationHelpers.create_mock_session

create_role_permissions = AuthorizationHelpers.create_role_permissions
assert_has_permission = AuthorizationHelpers.assert_has_permission
assert_authorized = AuthorizationHelpers.assert_authorized




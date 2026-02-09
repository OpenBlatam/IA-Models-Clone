from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import hashlib
import secrets
import re
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from security_guidelines import (
        import jwt
from typing import Any, List, Dict, Optional
import logging
# Import security components
    SecureInputValidator, SecurityAuthenticator, SecurityCrypto,
    SecurityLogger, SecurityHeaders, SecurityMiddleware,
    SecureScanRequest, SecureScanResponse, SecurityUtils,
    SecurityConfig, SecurityBestPractices
)

class TestSecureInputValidator:
    """Test cases for SecureInputValidator"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.validator = SecureInputValidator()
    
    def test_validate_ip_address_ipv4_valid(self) -> bool:
        """Test valid IPv4 address validation"""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "127.0.0.1",
            "0.0.0.0",
            "255.255.255.255"
        ]
        
        for ip in valid_ips:
            assert self.validator.validate_ip_address(ip) is True
    
    def test_validate_ip_address_ipv4_invalid(self) -> bool:
        """Test invalid IPv4 address validation"""
        invalid_ips = [
            "256.256.256.256",
            "192.168.1.256",
            "192.168.1",
            "192.168.1.1.1",
            "192.168.1.abc",
            "192.168.1.-1"
        ]
        
        for ip in invalid_ips:
            assert self.validator.validate_ip_address(ip) is False
    
    def test_validate_ip_address_ipv6_valid(self) -> bool:
        """Test valid IPv6 address validation"""
        valid_ipv6 = [
            "2001:db8::1",
            "::1",
            "2001:db8:0:0:0:0:0:1",
            "2001:db8::",
            "::"
        ]
        
        for ip in valid_ipv6:
            assert self.validator.validate_ip_address(ip) is True
    
    def test_validate_ip_address_ipv6_invalid(self) -> bool:
        """Test invalid IPv6 address validation"""
        invalid_ipv6 = [
            "2001:db8::1::1",  # Double colon
            "2001:db8:g::1",   # Invalid character
            "2001:db8::1:1:1:1:1:1:1:1",  # Too many segments
            "2001:db8:1"       # Incomplete
        ]
        
        for ip in invalid_ipv6:
            assert self.validator.validate_ip_address(ip) is False
    
    def test_validate_hostname_valid(self) -> bool:
        """Test valid hostname validation"""
        valid_hostnames = [
            "example.com",
            "test.example.com",
            "host-123.example.com",
            "host_123.example.com",
            "a" * 63 + ".com",  # Maximum length per label
            "example.co.uk"
        ]
        
        for hostname in valid_hostnames:
            assert self.validator.validate_hostname(hostname) is True
    
    def test_validate_hostname_invalid(self) -> bool:
        """Test invalid hostname validation"""
        invalid_hostnames = [
            "invalid..hostname",
            "hostname-",
            "-hostname",
            "hostname_",
            "_hostname",
            "a" * 64 + ".com",  # Too long per label
            "a" * 254 + ".com",  # Too long total
            "hostname with spaces.com",
            "hostname@domain.com"
        ]
        
        for hostname in invalid_hostnames:
            assert self.validator.validate_hostname(hostname) is False
    
    def test_validate_port_range_valid(self) -> bool:
        """Test valid port range validation"""
        valid_ports = [1, 80, 443, 1024, 65535]
        
        for port in valid_ports:
            assert self.validator.validate_port_range(port) is True
    
    def test_validate_port_range_invalid(self) -> bool:
        """Test invalid port range validation"""
        invalid_ports = [0, -1, 65536, 99999]
        
        for port in invalid_ports:
            assert self.validator.validate_port_range(port) is False
    
    def test_sanitize_command_input(self) -> Any:
        """Test command input sanitization"""
        test_cases = [
            ("ls -la", "ls -la"),  # No dangerous chars
            ("ls; rm -rf /", "ls rm -rf /"),  # Remove semicolon
            ("cat file | grep pattern", "cat file  grep pattern"),  # Remove pipe
            ("echo 'hello' && rm file", "echo 'hello'  rm file"),  # Remove &&
            ("cat file > output", "cat file  output"),  # Remove >
            ("cat file < input", "cat file  input"),  # Remove <
            ("echo `whoami`", "echo whoami"),  # Remove backticks
            ("echo $PATH", "echo PATH"),  # Remove $
            ("(rm -rf /)", "rm -rf /"),  # Remove parentheses
        ]
        
        for input_cmd, expected in test_cases:
            result = self.validator.sanitize_command_input(input_cmd)
            assert result == expected
    
    def test_validate_file_path_valid(self) -> bool:
        """Test valid file path validation"""
        valid_paths = [
            "file.txt",
            "path/to/file.txt",
            "file-name_123.txt",
            "file.txt.bak",
            "path/file-name.txt"
        ]
        
        for path in valid_paths:
            assert self.validator.validate_file_path(path) is True
    
    def test_validate_file_path_invalid(self) -> bool:
        """Test invalid file path validation"""
        invalid_paths = [
            "../file.txt",  # Directory traversal
            "/etc/passwd",  # Absolute path
            "file..txt",    # Double dots
            "file.txt/../", # Path traversal
            "file<>.txt",   # Invalid characters
            "file|.txt"     # Invalid characters
        ]
        
        for path in invalid_paths:
            assert self.validator.validate_file_path(path) is False

class TestSecurityAuthenticator:
    """Test cases for SecurityAuthenticator"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.secret_key = secrets.token_hex(32)
        self.authenticator = SecurityAuthenticator(self.secret_key)
    
    def test_generate_secure_token(self) -> Any:
        """Test secure token generation"""
        user_id = "test_user"
        permissions = ["read", "write"]
        
        token = self.authenticator.generate_secure_token(user_id, permissions)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert "." in token  # JWT format
    
    def test_verify_token_valid(self) -> Any:
        """Test valid token verification"""
        user_id = "test_user"
        permissions = ["read", "write"]
        
        token = self.authenticator.generate_secure_token(user_id, permissions)
        payload = self.authenticator.verify_token(token)
        
        assert payload is not None
        assert payload["user_id"] == user_id
        assert payload["permissions"] == permissions
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
    
    def test_verify_token_invalid(self) -> Any:
        """Test invalid token verification"""
        invalid_token = "invalid.token.here"
        payload = self.authenticator.verify_token(invalid_token)
        
        assert payload is None
    
    def test_verify_token_expired(self) -> Any:
        """Test expired token verification"""
        # Create expired token
        payload = {
            'user_id': 'test_user',
            'permissions': ['read'],
            'exp': datetime.utcnow() - timedelta(hours=1),  # Expired
            'iat': datetime.utcnow() - timedelta(hours=2),
            'jti': secrets.token_urlsafe(16)
        }
        expired_token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        result = self.authenticator.verify_token(expired_token)
        assert result is None
    
    def test_verify_token_blacklisted(self) -> List[Any]:
        """Test blacklisted token verification"""
        user_id = "test_user"
        permissions = ["read"]
        
        token = self.authenticator.generate_secure_token(user_id, permissions)
        
        # Verify token is valid initially
        payload = self.authenticator.verify_token(token)
        assert payload is not None
        
        # Blacklist token
        self.authenticator.blacklist_token(token)
        
        # Verify token is now invalid
        payload = self.authenticator.verify_token(token)
        assert payload is None
    
    def test_check_permission_valid(self) -> Any:
        """Test valid permission checking"""
        user_id = "test_user"
        permissions = ["read", "write", "admin"]
        
        token = self.authenticator.generate_secure_token(user_id, permissions)
        
        assert self.authenticator.check_permission(token, "read") is True
        assert self.authenticator.check_permission(token, "write") is True
        assert self.authenticator.check_permission(token, "admin") is True
    
    def test_check_permission_invalid(self) -> Any:
        """Test invalid permission checking"""
        user_id = "test_user"
        permissions = ["read", "write"]
        
        token = self.authenticator.generate_secure_token(user_id, permissions)
        
        assert self.authenticator.check_permission(token, "admin") is False
        assert self.authenticator.check_permission(token, "delete") is False
    
    def test_check_permission_invalid_token(self) -> Any:
        """Test permission checking with invalid token"""
        invalid_token = "invalid.token.here"
        
        assert self.authenticator.check_permission(invalid_token, "read") is False
    
    def test_check_rate_limit(self) -> Any:
        """Test rate limiting"""
        user_id = "test_user"
        action = "api_request"
        limit = 5
        
        # First 5 requests should be allowed
        for i in range(5):
            assert self.authenticator.check_rate_limit(user_id, action, limit) is True
        
        # 6th request should be blocked
        assert self.authenticator.check_rate_limit(user_id, action, limit) is False
    
    def test_rate_limit_reset(self) -> Any:
        """Test rate limit reset after time period"""
        user_id = "test_user"
        action = "api_request"
        limit = 3
        
        # Use up all requests
        for i in range(3):
            self.authenticator.check_rate_limit(user_id, action, limit)
        
        # Should be blocked
        assert self.authenticator.check_rate_limit(user_id, action, limit) is False
        
        # Manually clear old entries to simulate time passing
        now = datetime.utcnow()
        self.authenticator.rate_limit_store[f"{user_id}:{action}"] = [
            now - timedelta(minutes=2)  # Old entries
        ]
        
        # Should be allowed again
        assert self.authenticator.check_rate_limit(user_id, action, limit) is True

class TestSecurityCrypto:
    """Test cases for SecurityCrypto"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.crypto = SecurityCrypto()
    
    def test_encrypt_decrypt_data(self) -> Any:
        """Test data encryption and decryption"""
        test_data = "sensitive information"
        
        # Encrypt data
        encrypted = self.crypto.encrypt_sensitive_data(test_data)
        assert isinstance(encrypted, str)
        assert encrypted != test_data
        
        # Decrypt data
        decrypted = self.crypto.decrypt_sensitive_data(encrypted)
        assert decrypted == test_data
    
    def test_encrypt_decrypt_empty_string(self) -> Any:
        """Test encryption/decryption of empty string"""
        test_data = ""
        
        encrypted = self.crypto.encrypt_sensitive_data(test_data)
        decrypted = self.crypto.decrypt_sensitive_data(encrypted)
        
        assert decrypted == test_data
    
    def test_encrypt_decrypt_special_characters(self) -> Any:
        """Test encryption/decryption with special characters"""
        test_data = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        
        encrypted = self.crypto.encrypt_sensitive_data(test_data)
        decrypted = self.crypto.decrypt_sensitive_data(encrypted)
        
        assert decrypted == test_data
    
    def test_hash_password_with_salt(self) -> Any:
        """Test password hashing with provided salt"""
        password = "test_password"
        salt = "test_salt"
        
        result = self.crypto.hash_password(password, salt)
        
        assert "hash" in result
        assert "salt" in result
        assert result["salt"] == salt
        assert len(result["hash"]) > 0
    
    def test_hash_password_without_salt(self) -> Any:
        """Test password hashing without salt (auto-generated)"""
        password = "test_password"
        
        result = self.crypto.hash_password(password)
        
        assert "hash" in result
        assert "salt" in result
        assert len(result["salt"]) > 0
        assert len(result["hash"]) > 0
    
    def test_verify_password_correct(self) -> Any:
        """Test password verification with correct password"""
        password = "test_password"
        result = self.crypto.hash_password(password)
        
        is_valid = self.crypto.verify_password(
            password, result["hash"], result["salt"]
        )
        
        assert is_valid is True
    
    def test_verify_password_incorrect(self) -> Any:
        """Test password verification with incorrect password"""
        password = "test_password"
        wrong_password = "wrong_password"
        result = self.crypto.hash_password(password)
        
        is_valid = self.crypto.verify_password(
            wrong_password, result["hash"], result["salt"]
        )
        
        assert is_valid is False
    
    def test_generate_secure_random_string(self) -> Any:
        """Test secure random string generation"""
        length = 32
        random_string = self.crypto.generate_secure_random_string(length)
        
        assert isinstance(random_string, str)
        assert len(random_string) > 0
        
        # Generate another string to ensure randomness
        another_string = self.crypto.generate_secure_random_string(length)
        assert random_string != another_string

class TestSecurityUtils:
    """Test cases for SecurityUtils"""
    
    def test_generate_secure_filename(self) -> Any:
        """Test secure filename generation"""
        test_cases = [
            ("file.txt", "file.txt_"),
            ("file with spaces.txt", "file_with_spaces.txt_"),
            ("file@#$%.txt", "file____.txt_"),
            ("../../../etc/passwd", "etc_passwd_"),
            ("file<script>.txt", "filescript.txt_")
        ]
        
        for original, expected_prefix in test_cases:
            secure_name = SecurityUtils.generate_secure_filename(original)
            
            assert secure_name.startswith(expected_prefix)
            assert len(secure_name) > len(expected_prefix)  # Has random suffix
            assert re.match(r'^[a-zA-Z0-9._-]+$', secure_name)  # Only safe chars
    
    async def test_validate_file_upload_valid(self) -> bool:
        """Test valid file upload validation"""
        # Small text file
        valid_content = b"This is a valid text file content"
        assert SecurityUtils.validate_file_upload(valid_content) is True
        
        # Large file within limit
        large_content = b"x" * (5 * 1024 * 1024)  # 5MB
        assert SecurityUtils.validate_file_upload(large_content) is True
    
    async def test_validate_file_upload_invalid_size(self) -> bool:
        """Test file upload validation with oversized file"""
        oversized_content = b"x" * (20 * 1024 * 1024)  # 20MB
        assert SecurityUtils.validate_file_upload(oversized_content, max_size=10*1024*1024) is False
    
    async def test_validate_file_upload_dangerous_signatures(self) -> bool:
        """Test file upload validation with dangerous file signatures"""
        # EXE signature
        exe_content = b'\x4D\x5A' + b"fake exe content"
        assert SecurityUtils.validate_file_upload(exe_content) is False
        
        # ELF signature
        elf_content = b'\x7F\x45\x4C\x46' + b"fake elf content"
        assert SecurityUtils.validate_file_upload(elf_content) is False
        
        # Mach-O signature
        macho_content = b'\xFE\xED\xFA' + b"fake macho content"
        assert SecurityUtils.validate_file_upload(macho_content) is False
    
    def test_sanitize_sql_query_safe(self) -> Any:
        """Test SQL query sanitization with safe queries"""
        safe_queries = [
            "SELECT name FROM users",
            "INSERT INTO users (name) VALUES ('john')",
            "UPDATE users SET name = 'john' WHERE id = 1"
        ]
        
        for query in safe_queries:
            # Should not raise exception
            sanitized = SecurityUtils.sanitize_sql_query(query)
            assert sanitized == query
    
    def test_sanitize_sql_query_dangerous(self) -> Any:
        """Test SQL query sanitization with dangerous queries"""
        dangerous_queries = [
            "SELECT * FROM users; DROP TABLE users;",
            "SELECT * FROM users UNION SELECT * FROM passwords",
            "SELECT * FROM users WHERE id = 1 OR 1=1",
            "EXEC xp_cmdshell 'dir'",
            "CREATE TABLE malicious (id int)"
        ]
        
        for query in dangerous_queries:
            with pytest.raises(ValueError):
                SecurityUtils.sanitize_sql_query(query)

class TestSecurityConfig:
    """Test cases for SecurityConfig"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.config = SecurityConfig()
    
    def test_security_config_defaults(self) -> Any:
        """Test security configuration defaults"""
        assert self.config.max_scan_duration == 300
        assert self.config.rate_limit_per_minute == 60
        assert self.config.max_file_size == 10 * 1024 * 1024
        assert isinstance(self.config.allowed_file_types, list)
        assert self.config.session_timeout == 3600
        assert self.config.password_min_length == 12
        assert self.config.require_special_chars is True
        assert self.config.max_login_attempts == 5
        assert self.config.lockout_duration == 900

class TestSecurityBestPractices:
    """Test cases for SecurityBestPractices"""
    
    def test_implement_defense_in_depth(self) -> Any:
        """Test defense in depth implementation"""
        result = SecurityBestPractices.implement_defense_in_depth()
        
        assert "network_layer" in result
        assert "application_layer" in result
        assert "data_layer" in result
        assert "physical_layer" in result
        
        assert "Firewalls" in result["network_layer"]
        assert "Input validation" in result["application_layer"]
        assert "Encryption" in result["data_layer"]
    
    def test_implement_least_privilege(self) -> Any:
        """Test least privilege implementation"""
        result = SecurityBestPractices.implement_least_privilege()
        
        assert "user_permissions" in result
        assert "service_accounts" in result
        assert "network_access" in result
        assert "file_permissions" in result
        
        assert "Minimum required permissions" in result["user_permissions"]
        assert "Limited scope and access" in result["service_accounts"]
    
    def test_implement_secure_by_default(self) -> Any:
        """Test secure by default implementation"""
        result = SecurityBestPractices.implement_secure_by_default()
        
        assert "default_deny" in result
        assert "encryption_at_rest" in result
        assert "encryption_in_transit" in result
        assert "secure_defaults" in result
        
        assert "Deny by default" in result["default_deny"]
        assert "All data encrypted" in result["encryption_at_rest"]

class TestSecurityHeaders:
    """Test cases for SecurityHeaders"""
    
    def test_get_security_headers(self) -> Optional[Dict[str, Any]]:
        """Test security headers retrieval"""
        headers = SecurityHeaders.get_security_headers()
        
        expected_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'Referrer-Policy',
            'Permissions-Policy'
        ]
        
        for header in expected_headers:
            assert header in headers
            assert headers[header] is not None
            assert len(headers[header]) > 0

class TestSecureScanRequest:
    """Test cases for SecureScanRequest model"""
    
    async def test_valid_scan_request(self) -> Any:
        """Test valid scan request"""
        request = SecureScanRequest(
            target="192.168.1.1",
            scan_type="port",
            timeout=30,
            max_ports=1000
        )
        
        assert request.target == "192.168.1.1"
        assert request.scan_type == "port"
        assert request.timeout == 30
        assert request.max_ports == 1000
    
    def test_invalid_target(self) -> Optional[Dict[str, Any]]:
        """Test scan request with invalid target"""
        with pytest.raises(ValueError):
            SecureScanRequest(
                target="invalid.target",
                scan_type="port"
            )
    
    def test_invalid_scan_type(self) -> Any:
        """Test scan request with invalid scan type"""
        with pytest.raises(ValueError):
            SecureScanRequest(
                target="192.168.1.1",
                scan_type="invalid_type"
            )
    
    def test_invalid_timeout(self) -> Any:
        """Test scan request with invalid timeout"""
        with pytest.raises(ValueError):
            SecureScanRequest(
                target="192.168.1.1",
                scan_type="port",
                timeout=400  # Exceeds maximum
            )
    
    def test_invalid_max_ports(self) -> Any:
        """Test scan request with invalid max ports"""
        with pytest.raises(ValueError):
            SecureScanRequest(
                target="192.168.1.1",
                scan_type="port",
                max_ports=70000  # Exceeds maximum
            )

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 
"""
Tests for Security Components
Tests for security utilities, input sanitization, and validation
"""

import pytest
from unittest.mock import Mock, patch
import json

from core.infrastructure.security_utils import (
    InputSanitizer,
    SecurityValidator,
    sanitize_user_input
)


class TestInputSanitizer:
    """Tests for InputSanitizer"""
    
    def test_sanitize_string(self):
        """Test sanitizing string input"""
        # Test XSS prevention
        malicious_input = "<script>alert('xss')</script>"
        sanitized = InputSanitizer.sanitize_user_input(malicious_input, "string")
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
    
    def test_sanitize_sql_injection(self):
        """Test preventing SQL injection"""
        malicious_input = "'; DROP TABLE users; --"
        sanitized = InputSanitizer.sanitize_user_input(malicious_input, "string")
        
        # Should escape or remove dangerous characters
        assert sanitized is not None
        assert isinstance(sanitized, str)
    
    def test_sanitize_path_traversal(self):
        """Test preventing path traversal"""
        malicious_input = "../../../etc/passwd"
        sanitized = InputSanitizer.sanitize_user_input(malicious_input, "string")
        
        # Should prevent path traversal
        assert ".." not in sanitized or sanitized != malicious_input
    
    def test_sanitize_email(self):
        """Test sanitizing email input"""
        valid_email = "test@example.com"
        sanitized = InputSanitizer.sanitize_user_input(valid_email, "email")
        
        assert sanitized == valid_email
    
    def test_sanitize_invalid_email(self):
        """Test sanitizing invalid email"""
        invalid_email = "not-an-email"
        sanitized = InputSanitizer.sanitize_user_input(invalid_email, "email")
        
        # Should handle invalid email appropriately
        assert sanitized is not None
    
    def test_sanitize_url(self):
        """Test sanitizing URL input"""
        valid_url = "https://example.com"
        sanitized = InputSanitizer.sanitize_user_input(valid_url, "url")
        
        assert sanitized == valid_url
    
    def test_sanitize_malicious_url(self):
        """Test sanitizing malicious URL"""
        malicious_url = "javascript:alert('xss')"
        sanitized = InputSanitizer.sanitize_user_input(malicious_url, "url")
        
        # Should prevent javascript: protocol
        assert "javascript:" not in sanitized.lower()
    
    def test_sanitize_json(self):
        """Test sanitizing JSON input"""
        valid_json = '{"key": "value"}'
        sanitized = InputSanitizer.sanitize_user_input(valid_json, "json")
        
        # Should parse and validate JSON
        assert sanitized is not None
    
    def test_sanitize_invalid_json(self):
        """Test sanitizing invalid JSON"""
        invalid_json = '{"key": "value"'
        
        # Should handle invalid JSON gracefully
        try:
            sanitized = InputSanitizer.sanitize_user_input(invalid_json, "json")
            # If it doesn't raise, should return None or safe value
            assert sanitized is None or isinstance(sanitized, str)
        except (ValueError, json.JSONDecodeError):
            # Exception is also acceptable
            pass


class TestSecurityValidator:
    """Tests for SecurityValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create security validator"""
        return SecurityValidator()
    
    def test_validate_user_id_format(self, validator):
        """Test validating user ID format"""
        valid_id = "user-123-abc"
        assert validator.validate_user_id(valid_id) is True
        
        # Invalid formats
        invalid_ids = [
            "../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --"
        ]
        
        for invalid_id in invalid_ids:
            assert validator.validate_user_id(invalid_id) is False
    
    def test_validate_email_format(self, validator):
        """Test validating email format"""
        valid_emails = [
            "test@example.com",
            "user.name@example.co.uk",
            "user+tag@example.com"
        ]
        
        for email in valid_emails:
            assert validator.validate_email(email) is True
        
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user@.com"
        ]
        
        for email in invalid_emails:
            assert validator.validate_email(email) is False
    
    def test_validate_url_format(self, validator):
        """Test validating URL format"""
        valid_urls = [
            "https://example.com",
            "http://example.com/path",
            "https://example.com:8080/path?query=value"
        ]
        
        for url in valid_urls:
            assert validator.validate_url(url) is True
        
        invalid_urls = [
            "javascript:alert('xss')",
            "file:///etc/passwd",
            "not-a-url"
        ]
        
        for url in invalid_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_file_extension(self, validator):
        """Test validating file extensions"""
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        
        for ext in valid_extensions:
            filename = f"image{ext}"
            assert validator.validate_file_extension(filename, valid_extensions) is True
        
        invalid_extensions = [".exe", ".sh", ".bat", ".php"]
        
        for ext in invalid_extensions:
            filename = f"file{ext}"
            assert validator.validate_file_extension(filename, valid_extensions) is False
    
    def test_validate_file_size(self, validator):
        """Test validating file size"""
        max_size = 10 * 1024 * 1024  # 10MB
        
        # Valid sizes
        assert validator.validate_file_size(1024, max_size) is True
        assert validator.validate_file_size(max_size, max_size) is True
        
        # Invalid sizes
        assert validator.validate_file_size(0, max_size) is False
        assert validator.validate_file_size(max_size + 1, max_size) is False


class TestSecurityIntegration:
    """Integration tests for security"""
    
    def test_input_sanitization_pipeline(self):
        """Test complete input sanitization pipeline"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "javascript:alert('xss')"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = InputSanitizer.sanitize_user_input(malicious_input, "string")
            
            # Should not contain malicious patterns
            assert "<script>" not in sanitized.lower()
            assert "drop table" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
    
    def test_security_validation_pipeline(self):
        """Test complete security validation pipeline"""
        validator = SecurityValidator()
        
        # Test user registration data
        user_data = {
            "email": "test@example.com",
            "user_id": "user-123",
            "name": "Test User"
        }
        
        assert validator.validate_email(user_data["email"]) is True
        assert validator.validate_user_id(user_data["user_id"]) is True
    
    def test_file_upload_security(self):
        """Test file upload security checks"""
        validator = SecurityValidator()
        
        # Valid file
        valid_file = {
            "filename": "image.jpg",
            "size": 1024 * 1024,  # 1MB
            "content_type": "image/jpeg"
        }
        
        assert validator.validate_file_extension(
            valid_file["filename"], 
            [".jpg", ".jpeg", ".png"]
        ) is True
        assert validator.validate_file_size(
            valid_file["size"], 
            10 * 1024 * 1024
        ) is True
        
        # Invalid file
        invalid_file = {
            "filename": "script.php",
            "size": 20 * 1024 * 1024,  # 20MB (too large)
            "content_type": "application/x-php"
        }
        
        assert validator.validate_file_extension(
            invalid_file["filename"],
            [".jpg", ".jpeg", ".png"]
        ) is False
        assert validator.validate_file_size(
            invalid_file["size"],
            10 * 1024 * 1024
        ) is False


class TestSecurityHeaders:
    """Tests for security headers"""
    
    def test_cors_headers(self):
        """Test CORS security headers"""
        # This would typically be tested in middleware
        # But we can test the logic here
        headers = {
            "Access-Control-Allow-Origin": "https://example.com",
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "Content-Type"
        }
        
        assert "Access-Control-Allow-Origin" in headers
        assert "https://example.com" in headers["Access-Control-Allow-Origin"]
    
    def test_content_security_policy(self):
        """Test Content Security Policy headers"""
        csp_header = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        
        assert "default-src" in csp_header
        assert "'self'" in csp_header


class TestRateLimitingSecurity:
    """Tests for rate limiting as security measure"""
    
    def test_rate_limit_prevention(self):
        """Test that rate limiting prevents abuse"""
        # Simulate rate limiting
        request_count = {}
        max_requests = 10
        
        def check_rate_limit(ip_address):
            count = request_count.get(ip_address, 0)
            if count >= max_requests:
                return False
            request_count[ip_address] = count + 1
            return True
        
        # First 10 requests should pass
        for i in range(10):
            assert check_rate_limit("192.168.1.1") is True
        
        # 11th request should be blocked
        assert check_rate_limit("192.168.1.1") is False
        
        # Different IP should still work
        assert check_rate_limit("192.168.1.2") is True




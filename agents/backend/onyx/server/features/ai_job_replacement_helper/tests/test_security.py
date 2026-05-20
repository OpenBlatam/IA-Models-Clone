"""
Tests for Security Service
"""

import pytest
from core.security import SecurityService


@pytest.fixture
def security_service():
    return SecurityService()


def test_hash_password(security_service):
    """Test password hashing"""
    password = "test_password_123"
    hash1, salt1 = security_service.hash_password(password)
    hash2, salt2 = security_service.hash_password(password, salt1)
    
    assert hash1 == hash2
    assert salt1 is not None


def test_verify_password(security_service):
    """Test password verification"""
    password = "test_password_123"
    password_hash, salt = security_service.hash_password(password)
    
    assert security_service.verify_password(password, password_hash, salt) is True
    assert security_service.verify_password("wrong_password", password_hash, salt) is False


def test_generate_secure_token(security_service):
    """Test secure token generation"""
    token1 = security_service.generate_secure_token()
    token2 = security_service.generate_secure_token()
    
    assert len(token1) >= 32
    assert token1 != token2  # Should be unique


def test_sanitize_input(security_service):
    """Test input sanitization"""
    dangerous_input = '<script>alert("xss")</script>Hello'
    sanitized = security_service.sanitize_input(dangerous_input)
    
    assert "<script>" not in sanitized
    assert "alert" not in sanitized


def test_check_sql_injection(security_service):
    """Test SQL injection detection"""
    safe_input = "Hello world"
    dangerous_input = "SELECT * FROM users"
    
    assert security_service.check_sql_injection(safe_input) is False
    assert security_service.check_sql_injection(dangerous_input) is True


def test_check_xss(security_service):
    """Test XSS detection"""
    safe_input = "Hello world"
    dangerous_input = '<script>alert("xss")</script>'
    
    assert security_service.check_xss(safe_input) is False
    assert security_service.check_xss(dangerous_input) is True





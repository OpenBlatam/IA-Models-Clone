"""
Tests for AdvancedSecurity utility
"""

import pytest
from datetime import datetime, timedelta

from ..utils.advanced_security import AdvancedSecurity


class TestAdvancedSecurity:
    """Test suite for AdvancedSecurity"""

    def test_init(self):
        """Test AdvancedSecurity initialization"""
        security = AdvancedSecurity()
        assert security.failed_attempts == {}
        assert security.blocked_ips == {}
        assert security.api_keys == {}
        assert security.max_failed_attempts == 5

    def test_generate_api_key(self):
        """Test generating API key"""
        security = AdvancedSecurity()
        
        api_key = security.generate_api_key("user-123", ["read", "write"])
        
        assert api_key is not None
        assert len(api_key) > 0
        assert api_key in security.api_keys
        assert security.api_keys[api_key]["user_id"] == "user-123"

    def test_generate_api_key_with_expiration(self):
        """Test generating API key with expiration"""
        security = AdvancedSecurity()
        
        api_key = security.generate_api_key("user-456", ["read"], expires_in_days=7)
        
        assert api_key in security.api_keys
        assert security.api_keys[api_key]["expires_at"] is not None

    def test_validate_api_key(self):
        """Test validating API key"""
        security = AdvancedSecurity()
        
        api_key = security.generate_api_key("user-789", ["read", "write"])
        
        assert security.validate_api_key(api_key) is True
        assert security.validate_api_key("invalid-key") is False

    def test_validate_api_key_with_permission(self):
        """Test validating API key with permission"""
        security = AdvancedSecurity()
        
        api_key = security.generate_api_key("user-999", ["read"])
        
        assert security.validate_api_key(api_key, required_permission="read") is True
        assert security.validate_api_key(api_key, required_permission="write") is False

    def test_validate_expired_api_key(self):
        """Test validating expired API key"""
        security = AdvancedSecurity()
        
        # Create expired key (simulate)
        api_key = security.generate_api_key("user-expired", ["read"], expires_in_days=-1)
        
        # Should be invalid
        assert security.validate_api_key(api_key) is False

    def test_record_failed_attempt(self):
        """Test recording failed attempt"""
        security = AdvancedSecurity()
        
        security.record_failed_attempt("192.168.1.1")
        
        assert len(security.failed_attempts["192.168.1.1"]) == 1

    def test_block_ip_after_failed_attempts(self):
        """Test blocking IP after multiple failed attempts"""
        security = AdvancedSecurity()
        
        ip = "192.168.1.100"
        
        # Record multiple failed attempts
        for i in range(6):
            security.record_failed_attempt(ip)
        
        # Should be blocked
        assert security.is_ip_blocked(ip) is True

    def test_is_ip_blocked(self):
        """Test checking if IP is blocked"""
        security = AdvancedSecurity()
        
        ip = "192.168.1.200"
        
        # Not blocked initially
        assert security.is_ip_blocked(ip) is False
        
        # Block it
        security.blocked_ips[ip] = datetime.now()
        
        assert security.is_ip_blocked(ip) is True

    def test_unblock_ip(self):
        """Test unblocking IP"""
        security = AdvancedSecurity()
        
        ip = "192.168.1.300"
        security.blocked_ips[ip] = datetime.now()
        
        security.unblock_ip(ip)
        
        assert ip not in security.blocked_ips


"""
Tests refactorizados para helpers de rate limiting
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import time
from api.utils.rate_limit_helpers import (
    check_rate_limit,
    get_rate_limit_info
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestCheckRateLimitRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para check_rate_limit"""
    
    def test_check_rate_limit_allowed(self):
        """Test de request permitido"""
        identifier = "test-user-1"
        is_allowed, retry_after = check_rate_limit(
            identifier,
            max_requests=10,
            window_seconds=60
        )
        
        assert is_allowed is True
        assert retry_after is None
    
    def test_check_rate_limit_exceeded(self):
        """Test de exceder rate limit"""
        identifier = "test-user-2"
        
        # Hacer múltiples requests
        for _ in range(5):
            check_rate_limit(identifier, max_requests=5, window_seconds=60)
        
        # El siguiente debería ser rechazado
        is_allowed, retry_after = check_rate_limit(
            identifier,
            max_requests=5,
            window_seconds=60
        )
        
        assert is_allowed is False
        assert retry_after is not None
        assert retry_after > 0
    
    def test_check_rate_limit_different_identifiers(self):
        """Test de diferentes identificadores"""
        id1 = "user-1"
        id2 = "user-2"
        
        # Llenar rate limit para user-1
        for _ in range(5):
            check_rate_limit(id1, max_requests=5, window_seconds=60)
        
        # user-2 debería estar permitido
        is_allowed, _ = check_rate_limit(id2, max_requests=5, window_seconds=60)
        
        assert is_allowed is True
    
    def test_check_rate_limit_window_expires(self):
        """Test de que la ventana expira"""
        identifier = "test-user-3"
        
        # Llenar rate limit
        for _ in range(5):
            check_rate_limit(identifier, max_requests=5, window_seconds=1)
        
        # Esperar a que expire la ventana
        time.sleep(1.1)
        
        # Debería estar permitido de nuevo
        is_allowed, _ = check_rate_limit(identifier, max_requests=5, window_seconds=1)
        
        assert is_allowed is True


class TestGetRateLimitInfoRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_rate_limit_info"""
    
    def test_get_rate_limit_info_no_requests(self):
        """Test de info sin requests"""
        identifier = "test-user-4"
        info = get_rate_limit_info(identifier, max_requests=10, window_seconds=60)
        
        assert info["remaining"] == 10
        assert info["limit"] == 10
        assert info["reset_in"] == 60
    
    def test_get_rate_limit_info_with_requests(self):
        """Test de info con requests"""
        identifier = "test-user-5"
        
        # Hacer algunos requests
        for _ in range(3):
            check_rate_limit(identifier, max_requests=10, window_seconds=60)
        
        info = get_rate_limit_info(identifier, max_requests=10, window_seconds=60)
        
        assert info["remaining"] < 10
        assert info["limit"] == 10
        assert "reset_in" in info




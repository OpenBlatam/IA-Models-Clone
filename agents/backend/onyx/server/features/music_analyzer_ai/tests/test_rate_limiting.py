"""
Tests de rate limiting avanzado
"""

import pytest
from unittest.mock import Mock, patch
import time
from collections import defaultdict


class TestRateLimiting:
    """Tests de rate limiting"""
    
    def test_simple_rate_limit(self):
        """Test de rate limiting simple"""
        class SimpleRateLimiter:
            def __init__(self, max_requests, window_seconds):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = defaultdict(list)
            
            def is_allowed(self, identifier):
                now = time.time()
                user_requests = self.requests[identifier]
                
                # Limpiar requests antiguos
                user_requests[:] = [req_time for req_time in user_requests 
                                  if now - req_time < self.window_seconds]
                
                if len(user_requests) >= self.max_requests:
                    return False
                
                user_requests.append(now)
                return True
        
        limiter = SimpleRateLimiter(max_requests=5, window_seconds=60)
        
        # Permitir 5 requests
        for i in range(5):
            assert limiter.is_allowed("user1") == True
        
        # El 6to debe ser bloqueado
        assert limiter.is_allowed("user1") == False
    
    def test_rate_limit_per_user(self):
        """Test de rate limiting por usuario"""
        class PerUserRateLimiter:
            def __init__(self, max_requests=10, window_seconds=60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = defaultdict(list)
            
            def is_allowed(self, user_id):
                now = time.time()
                user_requests = self.requests[user_id]
                
                # Limpiar requests antiguos
                user_requests[:] = [req_time for req_time in user_requests 
                                  if now - req_time < self.window_seconds]
                
                if len(user_requests) >= self.max_requests:
                    return False
                
                user_requests.append(now)
                return True
        
        limiter = PerUserRateLimiter(max_requests=3, window_seconds=60)
        
        # Usuario 1 puede hacer 3 requests
        for i in range(3):
            assert limiter.is_allowed("user1") == True
        
        assert limiter.is_allowed("user1") == False
        
        # Usuario 2 puede hacer sus propios 3 requests
        for i in range(3):
            assert limiter.is_allowed("user2") == True
    
    def test_rate_limit_with_retry_after(self):
        """Test de rate limiting con retry-after"""
        class RateLimiterWithRetry:
            def __init__(self, max_requests=10, window_seconds=60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = defaultdict(list)
            
            def check_rate_limit(self, identifier):
                now = time.time()
                user_requests = self.requests[identifier]
                
                # Limpiar requests antiguos
                user_requests[:] = [req_time for req_time in user_requests 
                                  if now - req_time < self.window_seconds]
                
                if len(user_requests) >= self.max_requests:
                    # Calcular cuándo puede hacer el siguiente request
                    oldest_request = min(user_requests)
                    retry_after = int(self.window_seconds - (now - oldest_request))
                    return {
                        "allowed": False,
                        "retry_after": retry_after,
                        "remaining": 0
                    }
                
                user_requests.append(now)
                return {
                    "allowed": True,
                    "retry_after": 0,
                    "remaining": self.max_requests - len(user_requests)
                }
        
        limiter = RateLimiterWithRetry(max_requests=2, window_seconds=60)
        
        result1 = limiter.check_rate_limit("user1")
        assert result1["allowed"] == True
        
        result2 = limiter.check_rate_limit("user1")
        assert result2["allowed"] == True
        
        result3 = limiter.check_rate_limit("user1")
        assert result3["allowed"] == False
        assert result3["retry_after"] > 0


class TestSlidingWindowRateLimit:
    """Tests de rate limiting con ventana deslizante"""
    
    def test_sliding_window(self):
        """Test de ventana deslizante"""
        class SlidingWindowLimiter:
            def __init__(self, max_requests, window_seconds):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = defaultdict(list)
            
            def is_allowed(self, identifier):
                now = time.time()
                user_requests = self.requests[identifier]
                
                # Eliminar requests fuera de la ventana
                cutoff = now - self.window_seconds
                user_requests[:] = [req_time for req_time in user_requests 
                                  if req_time > cutoff]
                
                if len(user_requests) >= self.max_requests:
                    return False
                
                user_requests.append(now)
                return True
        
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=1)
        
        # Hacer 3 requests
        for i in range(3):
            assert limiter.is_allowed("user1") == True
        
        # El 4to debe ser bloqueado
        assert limiter.is_allowed("user1") == False
        
        # Esperar y debería permitir de nuevo
        time.sleep(1.1)
        assert limiter.is_allowed("user1") == True


class TestTokenBucket:
    """Tests de token bucket rate limiting"""
    
    def test_token_bucket(self):
        """Test de token bucket"""
        class TokenBucket:
            def __init__(self, capacity, refill_rate):
                self.capacity = capacity
                self.refill_rate = refill_rate  # tokens por segundo
                self.tokens = capacity
                self.last_refill = time.time()
            
            def consume(self, tokens=1):
                now = time.time()
                # Refill tokens
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, 
                                self.tokens + elapsed * self.refill_rate)
                self.last_refill = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                return False
        
        bucket = TokenBucket(capacity=5, refill_rate=1)  # 1 token por segundo
        
        # Consumir 5 tokens
        for i in range(5):
            assert bucket.consume() == True
        
        # No hay más tokens
        assert bucket.consume() == False
        
        # Esperar 1 segundo y debería tener 1 token
        time.sleep(1.1)
        assert bucket.consume() == True


class TestRateLimitHeaders:
    """Tests de headers de rate limiting"""
    
    def test_rate_limit_headers(self):
        """Test de headers de rate limiting"""
        def get_rate_limit_headers(limiter, identifier):
            now = time.time()
            user_requests = limiter.requests.get(identifier, [])
            
            # Limpiar requests antiguos
            user_requests[:] = [req_time for req_time in user_requests 
                              if now - req_time < limiter.window_seconds]
            
            remaining = max(0, limiter.max_requests - len(user_requests))
            
            return {
                "X-RateLimit-Limit": str(limiter.max_requests),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(now + limiter.window_seconds))
            }
        
        class Limiter:
            def __init__(self):
                self.max_requests = 10
                self.window_seconds = 60
                self.requests = defaultdict(list)
        
        limiter = Limiter()
        limiter.requests["user1"] = [time.time() - 10] * 3  # 3 requests hace 10 seg
        
        headers = get_rate_limit_headers(limiter, "user1")
        
        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "7"
        assert "X-RateLimit-Reset" in headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


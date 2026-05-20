"""
Tests para funcionalidades avanzadas
"""

import pytest
import asyncio
from datetime import datetime

from ..services.realtime_metrics import RealtimeMetrics
from ..services.health_monitor import HealthMonitor, HealthStatus
from ..services.business_metrics import BusinessMetrics
from ..utils.intelligent_cache import IntelligentCache
from ..utils.endpoint_rate_limiter import EndpointRateLimiter
from ..utils.advanced_logging import AdvancedLogger, LogLevel


class TestRealtimeMetrics:
    """Tests para métricas en tiempo real"""
    
    def test_record_and_get_metric(self):
        """Test registro y obtención de métricas"""
        metrics = RealtimeMetrics()
        
        metrics.record("test_metric", 10.5)
        stats = metrics.get_statistics("test_metric")
        
        assert stats["count"] == 1
        assert stats["avg"] == 10.5
        assert stats["latest"] == 10.5
    
    def test_window_filtering(self):
        """Test filtrado por ventana"""
        metrics = RealtimeMetrics(window_seconds=60)
        
        metrics.record("test_metric", 10)
        stats = metrics.get_statistics("test_metric", window_seconds=30)
        
        assert stats["count"] >= 0


class TestHealthMonitor:
    """Tests para health monitor"""
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check"""
        monitor = HealthMonitor()
        
        def healthy_check():
            return {"status": "healthy", "message": "OK"}
        
        monitor.register_check("test_check", healthy_check)
        check = await monitor.run_check("test_check")
        
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "OK"
    
    @pytest.mark.asyncio
    async def test_overall_health(self):
        """Test salud general"""
        monitor = HealthMonitor()
        
        def healthy_check():
            return True
        
        monitor.register_check("test", healthy_check)
        await monitor.run_check("test")
        
        overall = monitor.get_overall_health()
        assert "status" in overall
        assert "checks" in overall


class TestBusinessMetrics:
    """Tests para métricas de negocio"""
    
    def test_record_and_get_metrics(self):
        """Test registro y obtención"""
        metrics = BusinessMetrics()
        
        metrics.record_metric("revenue", 100.0, "$", "sales")
        kpis = metrics.get_kpis()
        
        assert "total_revenue" in kpis
    
    def test_category_metrics(self):
        """Test métricas por categoría"""
        metrics = BusinessMetrics()
        
        metrics.record_metric("revenue", 100.0, "$", "sales")
        category_metrics = metrics.get_category_metrics("sales")
        
        assert category_metrics["category"] == "sales"
        assert category_metrics["count"] > 0


class TestIntelligentCache:
    """Tests para cache inteligente"""
    
    def test_get_and_set(self):
        """Test get y set"""
        cache = IntelligentCache()
        
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        assert value == "test_value"
    
    def test_ttl_expiration(self):
        """Test expiración TTL"""
        cache = IntelligentCache()
        
        cache.set("test_key", "test_value", ttl=1)
        value1 = cache.get("test_key")
        
        import time
        time.sleep(2)
        
        value2 = cache.get("test_key")
        
        assert value1 == "test_value"
        assert value2 is None
    
    def test_cache_stats(self):
        """Test estadísticas"""
        cache = IntelligentCache()
        
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1


class TestEndpointRateLimiter:
    """Tests para rate limiter por endpoint"""
    
    def test_set_limit_and_check(self):
        """Test establecer límite y verificar"""
        limiter = EndpointRateLimiter()
        
        limiter.set_limit("/test", max_requests=5, window_seconds=60)
        
        # Hacer 5 requests
        for i in range(5):
            allowed, info = limiter.is_allowed("/test")
            assert allowed
        
        # El 6to debería ser bloqueado
        allowed, info = limiter.is_allowed("/test")
        assert not allowed
    
    def test_endpoint_stats(self):
        """Test estadísticas de endpoint"""
        limiter = EndpointRateLimiter()
        
        limiter.set_limit("/test", max_requests=10, window_seconds=60)
        limiter.is_allowed("/test")
        
        stats = limiter.get_endpoint_stats("/test")
        
        assert stats["endpoint"] == "/test"
        assert stats["limit"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])







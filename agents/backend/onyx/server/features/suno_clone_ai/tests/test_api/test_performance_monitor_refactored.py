"""
Tests refactorizados para monitor de rendimiento
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import asyncio
import time
from api.utils.performance_monitor import (
    measure_time,
    performance_monitor,
    get_performance_stats,
    clear_performance_stats
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestMeasureTimeRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para measure_time context manager"""
    
    def test_measure_time_basic(self):
        """Test básico de medida de tiempo"""
        with measure_time("test_operation"):
            time.sleep(0.01)
        
        stats = get_performance_stats("test_operation")
        
        assert stats is not None
        assert stats["count"] == 1
        assert stats["total_time"] > 0
    
    def test_measure_time_multiple_calls(self):
        """Test de múltiples llamadas"""
        for _ in range(3):
            with measure_time("test_operation_2"):
                time.sleep(0.01)
        
        stats = get_performance_stats("test_operation_2")
        
        assert stats["count"] == 3
        assert stats["total_time"] > 0
        assert stats["min_time"] > 0
        assert stats["max_time"] > 0
    
    def test_measure_time_with_exception(self):
        """Test de medida de tiempo con excepción"""
        try:
            with measure_time("test_operation_3"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        stats = get_performance_stats("test_operation_3")
        
        # Debería registrar el tiempo incluso con excepción
        assert stats is not None


class TestPerformanceMonitorRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para performance_monitor decorator"""
    
    @pytest.mark.asyncio
    async def test_performance_monitor_async(self):
        """Test de decorator con función async"""
        @performance_monitor("async_test")
        async def async_func():
            await asyncio.sleep(0.01)
            return "result"
        
        result = await async_func()
        
        assert result == "result"
        
        stats = get_performance_stats("async_test")
        assert stats is not None
        assert stats["count"] == 1
    
    def test_performance_monitor_sync(self):
        """Test de decorator con función sync"""
        @performance_monitor("sync_test")
        def sync_func():
            time.sleep(0.01)
            return "result"
        
        result = sync_func()
        
        assert result == "result"
        
        stats = get_performance_stats("sync_test")
        assert stats is not None
        assert stats["count"] == 1


class TestGetPerformanceStatsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_performance_stats"""
    
    def test_get_performance_stats_existing(self):
        """Test de obtención de stats existentes"""
        with measure_time("stats_test"):
            time.sleep(0.01)
        
        stats = get_performance_stats("stats_test")
        
        assert stats is not None
        assert "count" in stats
        assert "total_time" in stats
        assert "min_time" in stats
        assert "max_time" in stats
        assert "avg_time" in stats
    
    def test_get_performance_stats_nonexistent(self):
        """Test de obtención de stats inexistentes"""
        stats = get_performance_stats("nonexistent_operation")
        
        assert stats is None
    
    def test_get_performance_stats_all(self):
        """Test de obtención de todas las stats"""
        with measure_time("op1"):
            pass
        with measure_time("op2"):
            pass
        
        all_stats = get_performance_stats()
        
        assert isinstance(all_stats, dict)
        assert "op1" in all_stats
        assert "op2" in all_stats


class TestClearPerformanceStatsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para clear_performance_stats"""
    
    def test_clear_performance_stats(self):
        """Test de limpieza de stats"""
        with measure_time("clear_test"):
            pass
        
        stats_before = get_performance_stats("clear_test")
        assert stats_before is not None
        
        clear_performance_stats()
        
        stats_after = get_performance_stats("clear_test")
        assert stats_after is None
    
    def test_clear_performance_stats_specific(self):
        """Test de limpieza de stats específicas"""
        with measure_time("op1"):
            pass
        with measure_time("op2"):
            pass
        
        clear_performance_stats("op1")
        
        assert get_performance_stats("op1") is None
        assert get_performance_stats("op2") is not None




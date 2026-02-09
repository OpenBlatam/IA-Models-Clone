"""
Tests para helpers de performance
"""

import pytest
import time
from tests.helpers.advanced_helpers import PerformanceHelper


class TestPerformanceHelper:
    """Tests para PerformanceHelper"""
    
    def test_measure_time_decorator(self):
        """Test del decorador measure_time"""
        @PerformanceHelper.measure_time
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        assert hasattr(slow_function, "elapsed_time")
        assert slow_function.elapsed_time >= 0.1
        assert slow_function.elapsed_time < 0.2
    
    def test_assert_performance_success(self):
        """Test de assert_performance con tiempo válido"""
        PerformanceHelper.assert_performance(0.05, 0.1, "test operation")
        # No debe lanzar excepción
    
    def test_assert_performance_failure(self):
        """Test de assert_performance con tiempo inválido"""
        with pytest.raises(AssertionError):
            PerformanceHelper.assert_performance(0.2, 0.1, "test operation")


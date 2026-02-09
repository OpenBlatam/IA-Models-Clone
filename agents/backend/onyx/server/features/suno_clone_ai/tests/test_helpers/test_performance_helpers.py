"""
Tests para helpers de performance
"""

import pytest
import asyncio
import time
from tests.helpers.advanced_helpers import PerformanceHelper, AsyncTestHelper


class TestPerformanceHelper:
    """Tests para PerformanceHelper"""
    
    @pytest.mark.unit
    def test_measure_execution_time_sync(self):
        """Test de medición de tiempo síncrono"""
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        execution_time = PerformanceHelper.measure_execution_time(slow_function)
        
        assert execution_time >= 0.1
        assert execution_time < 0.2  # Con margen de error
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_measure_execution_time_async(self):
        """Test de medición de tiempo asíncrono"""
        async def slow_async_function():
            await asyncio.sleep(0.1)
            return "done"
        
        execution_time = await PerformanceHelper.measure_async_execution_time(
            slow_async_function
        )
        
        assert execution_time >= 0.1
        assert execution_time < 0.2
    
    @pytest.mark.unit
    def test_assert_execution_time_under_success(self):
        """Test de aserción de tiempo exitosa"""
        def fast_function():
            time.sleep(0.01)
            return "done"
        
        PerformanceHelper.assert_execution_time_under(
            fast_function,
            max_time=0.1,
            message="Function should be fast"
        )
        # No debe lanzar excepción
    
    @pytest.mark.unit
    def test_assert_execution_time_under_failure(self):
        """Test de aserción de tiempo fallida"""
        def slow_function():
            time.sleep(0.2)
            return "done"
        
        with pytest.raises(AssertionError):
            PerformanceHelper.assert_execution_time_under(
                slow_function,
                max_time=0.1,
                message="Function is too slow"
            )


class TestAsyncTestHelper:
    """Tests para AsyncTestHelper"""
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_wait_for_condition_success(self):
        """Test de espera de condición exitosa"""
        condition_met = False
        
        async def set_condition():
            await asyncio.sleep(0.1)
            nonlocal condition_met
            condition_met = True
        
        # Ejecutar en background
        asyncio.create_task(set_condition())
        
        # Esperar condición
        result = await AsyncTestHelper.wait_for_condition(
            lambda: condition_met,
            timeout=1.0,
            interval=0.05
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_wait_for_condition_timeout(self):
        """Test de timeout en espera de condición"""
        with pytest.raises(TimeoutError):
            await AsyncTestHelper.wait_for_condition(
                lambda: False,  # Nunca se cumple
                timeout=0.1,
                interval=0.01,
                error_message="Condition never met"
            )
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_retry_async_success(self):
        """Test de retry exitoso"""
        attempts = 0
        
        async def flaky_function():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Temporary error")
            return "success"
        
        result = await AsyncTestHelper.retry_async(
            flaky_function,
            max_attempts=3,
            delay=0.01,
            exceptions=(ValueError,)
        )
        
        assert result == "success"
        assert attempts == 2
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_retry_async_max_attempts(self):
        """Test de retry con máximo de intentos"""
        async def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await AsyncTestHelper.retry_async(
                always_fails,
                max_attempts=3,
                delay=0.01,
                exceptions=(ValueError,)
            )


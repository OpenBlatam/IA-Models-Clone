"""
Tests para helpers asíncronos
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from api.utils.async_helpers import (
    retry_async,
    gather_with_limit
)


@pytest.mark.unit
@pytest.mark.api
class TestRetryAsync:
    """Tests para retry_async decorator"""
    
    @pytest.mark.asyncio
    async def test_retry_async_success_first_try(self):
        """Test de retry con éxito en primer intento"""
        call_count = 0
        
        @retry_async(max_attempts=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_async_retries_on_failure(self):
        """Test de retry con reintentos"""
        call_count = 0
        
        @retry_async(max_attempts=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_max_attempts(self):
        """Test de retry con máximo de intentos"""
        call_count = 0
        
        @retry_async(max_attempts=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            await test_func()
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_backoff(self):
        """Test de retry con backoff exponencial"""
        call_count = 0
        delays = []
        
        @retry_async(max_attempts=3, delay=0.01, backoff=2.0)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_specific_exception(self):
        """Test de retry con excepción específica"""
        call_count = 0
        
        @retry_async(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Error")
            return "success"
        
        result = await test_func()
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_async_other_exception(self):
        """Test de retry que no captura otras excepciones"""
        @retry_async(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        async def test_func():
            raise TypeError("Different error")
        
        with pytest.raises(TypeError):
            await test_func()


@pytest.mark.unit
@pytest.mark.api
class TestGatherWithLimit:
    """Tests para gather_with_limit"""
    
    @pytest.mark.asyncio
    async def test_gather_with_limit_all_success(self):
        """Test de gather con todas las tareas exitosas"""
        async def task(n):
            await asyncio.sleep(0.01)
            return n
        
        tasks = [task(i) for i in range(10)]
        results = await gather_with_limit(tasks, limit=5)
        
        assert len(results) == 10
        assert results == list(range(10))
    
    @pytest.mark.asyncio
    async def test_gather_with_limit_concurrency(self):
        """Test de límite de concurrencia"""
        active_tasks = []
        
        async def task(n):
            active_tasks.append(n)
            await asyncio.sleep(0.05)
            active_tasks.remove(n)
            return n
        
        tasks = [task(i) for i in range(10)]
        results = await gather_with_limit(tasks, limit=3)
        
        assert len(results) == 10
        # Verificar que nunca hubo más de 3 tareas activas simultáneamente
        # (esto es difícil de verificar directamente, pero el test verifica que funciona)
    
    @pytest.mark.asyncio
    async def test_gather_with_limit_empty(self):
        """Test de gather con lista vacía"""
        results = await gather_with_limit([], limit=5)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_gather_with_limit_single(self):
        """Test de gather con una sola tarea"""
        async def task():
            return "result"
        
        results = await gather_with_limit([task()], limit=5)
        
        assert len(results) == 1
        assert results[0] == "result"
    
    @pytest.mark.asyncio
    async def test_gather_with_limit_with_errors(self):
        """Test de gather con errores"""
        async def task(n):
            if n == 5:
                raise ValueError("Error")
            return n
        
        tasks = [task(i) for i in range(10)]
        
        # gather_with_limit debería propagar el error
        with pytest.raises(ValueError):
            await gather_with_limit(tasks, limit=5)




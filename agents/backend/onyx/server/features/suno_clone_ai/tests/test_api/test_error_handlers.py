"""
Tests para handlers de errores
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException, status

from api.utils.error_handlers import (
    handle_service_error,
    safe_execute,
    safe_execute_async
)


@pytest.mark.unit
@pytest.mark.api
class TestHandleServiceError:
    """Tests para handle_service_error"""
    
    def test_handle_service_error_http_exception(self):
        """Test de manejo de HTTPException"""
        error = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found"
        )
        
        result = handle_service_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == 404
    
    def test_handle_service_error_value_error(self):
        """Test de manejo de ValueError"""
        error = ValueError("Invalid input")
        
        result = handle_service_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_handle_service_error_generic_exception(self):
        """Test de manejo de excepción genérica"""
        error = Exception("Generic error")
        
        result = handle_service_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.unit
@pytest.mark.api
class TestSafeExecute:
    """Tests para safe_execute"""
    
    def test_safe_execute_success(self):
        """Test de ejecución exitosa"""
        def test_func():
            return "success"
        
        result = safe_execute(test_func)
        
        assert result == "success"
    
    def test_safe_execute_with_args(self):
        """Test de ejecución con argumentos"""
        def test_func(a, b):
            return a + b
        
        result = safe_execute(test_func, 1, 2)
        
        assert result == 3
    
    def test_safe_execute_with_kwargs(self):
        """Test de ejecución con kwargs"""
        def test_func(a, b=0):
            return a + b
        
        result = safe_execute(test_func, 1, b=2)
        
        assert result == 3
    
    def test_safe_execute_handles_error(self):
        """Test de manejo de error"""
        def test_func():
            raise ValueError("Error")
        
        result = safe_execute(test_func)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_safe_execute_default_value(self):
        """Test con valor por defecto"""
        def test_func():
            raise ValueError("Error")
        
        result = safe_execute(test_func, default="default_value")
        
        assert result == "default_value"


@pytest.mark.unit
@pytest.mark.api
class TestSafeExecuteAsync:
    """Tests para safe_execute_async"""
    
    @pytest.mark.asyncio
    async def test_safe_execute_async_success(self):
        """Test de ejecución async exitosa"""
        async def test_func():
            return "success"
        
        result = await safe_execute_async(test_func)
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_safe_execute_async_with_args(self):
        """Test de ejecución async con argumentos"""
        async def test_func(a, b):
            return a + b
        
        result = await safe_execute_async(test_func, 1, 2)
        
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_safe_execute_async_handles_error(self):
        """Test de manejo de error async"""
        async def test_func():
            raise ValueError("Error")
        
        result = await safe_execute_async(test_func)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
    
    @pytest.mark.asyncio
    async def test_safe_execute_async_default_value(self):
        """Test async con valor por defecto"""
        async def test_func():
            raise ValueError("Error")
        
        result = await safe_execute_async(test_func, default="default_value")
        
        assert result == "default_value"




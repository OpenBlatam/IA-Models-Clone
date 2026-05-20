"""
Tests for Graceful Degradation
Tests for fallback mechanisms and service degradation
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

from core.infrastructure.graceful_degradation import (
    GracefulDegradation,
    ServicePriority,
    graceful_degradation_decorator
)


class TestGracefulDegradation:
    """Tests for GracefulDegradation"""
    
    @pytest.fixture
    def degradation(self):
        """Create graceful degradation instance"""
        return GracefulDegradation()
    
    @pytest.mark.asyncio
    async def test_register_service(self, degradation):
        """Test registering a service"""
        fallback = Mock(return_value="fallback_result")
        
        degradation.register_service(
            service_name="test_service",
            priority=ServicePriority.HIGH,
            fallback=fallback
        )
        
        assert "test_service" in degradation.service_priorities
        assert degradation.service_priorities["test_service"] == ServicePriority.HIGH
        assert "test_service" in degradation.fallback_handlers
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, degradation):
        """Test executing with fallback when primary succeeds"""
        async def primary_func(arg):
            return f"primary-{arg}"
        
        result = await degradation.execute_with_fallback(
            service_name="test_service",
            primary_func=primary_func,
            "test"
        )
        
        assert result == "primary-test"
        assert degradation.service_status.get("test_service", True)
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_primary_fails(self, degradation):
        """Test executing with fallback when primary fails"""
        async def primary_func(arg):
            raise Exception("Primary failed")
        
        fallback = AsyncMock(return_value="fallback_result")
        degradation.register_service(
            service_name="test_service",
            priority=ServicePriority.MEDIUM,
            fallback=fallback
        )
        
        result = await degradation.execute_with_fallback(
            service_name="test_service",
            primary_func=primary_func,
            "test"
        )
        
        assert result == "fallback_result"
        assert degradation.service_status["test_service"] is False
        fallback.assert_called_once_with("test")
    
    @pytest.mark.asyncio
    async def test_critical_service_no_fallback(self, degradation):
        """Test that critical services don't use fallback"""
        async def primary_func(arg):
            raise Exception("Critical service failed")
        
        degradation.register_service(
            service_name="critical_service",
            priority=ServicePriority.CRITICAL
        )
        
        with pytest.raises(Exception):
            await degradation.execute_with_fallback(
                service_name="critical_service",
                primary_func=primary_func,
                "test"
            )
    
    @pytest.mark.asyncio
    async def test_fallback_also_fails(self, degradation):
        """Test when both primary and fallback fail"""
        async def primary_func(arg):
            raise Exception("Primary failed")
        
        async def fallback_func(arg):
            raise Exception("Fallback also failed")
        
        degradation.register_service(
            service_name="test_service",
            priority=ServicePriority.MEDIUM,
            fallback=fallback_func
        )
        
        with pytest.raises(Exception):
            await degradation.execute_with_fallback(
                service_name="test_service",
                primary_func=primary_func,
                "test"
            )
    
    @pytest.mark.asyncio
    async def test_get_service_status(self, degradation):
        """Test getting service status"""
        degradation.register_service("service1", ServicePriority.HIGH)
        degradation.service_status["service1"] = True
        
        status = degradation.get_service_status("service1")
        
        assert status is True
    
    @pytest.mark.asyncio
    async def test_get_all_statuses(self, degradation):
        """Test getting all service statuses"""
        degradation.register_service("service1", ServicePriority.HIGH)
        degradation.register_service("service2", ServicePriority.MEDIUM)
        
        statuses = degradation.get_all_statuses()
        
        assert "service1" in statuses
        assert "service2" in statuses


class TestGracefulDegradationDecorator:
    """Tests for graceful_degradation_decorator"""
    
    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator when function succeeds"""
        degradation = GracefulDegradation()
        
        @graceful_degradation_decorator(
            degradation,
            service_name="test_service",
            priority=ServicePriority.MEDIUM
        )
        async def test_function(arg):
            return f"result-{arg}"
        
        result = await test_function("test")
        
        assert result == "result-test"
    
    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self):
        """Test decorator with fallback"""
        degradation = GracefulDegradation()
        
        fallback = AsyncMock(return_value="fallback_result")
        degradation.register_service(
            service_name="test_service",
            priority=ServicePriority.MEDIUM,
            fallback=fallback
        )
        
        @graceful_degradation_decorator(
            degradation,
            service_name="test_service",
            priority=ServicePriority.MEDIUM
        )
        async def failing_function(arg):
            raise Exception("Function failed")
        
        result = await failing_function("test")
        
        assert result == "fallback_result"
        fallback.assert_called_once_with("test")




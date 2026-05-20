"""
Tests for Managers
Tests for RouterManager, ServiceLocator, and other managers
"""

import pytest
from unittest.mock import Mock
from fastapi import APIRouter

from api.routers.router_manager import RouterManager, get_router_manager
from api.services_locator import ServiceLocator, get_service_locator
from api.service_registry import ServiceRegistry
from core.feature_flags import FeatureFlagManager


class TestRouterManager:
    """Tests for RouterManager"""
    
    @pytest.fixture
    def router_manager(self):
        """Create router manager"""
        return RouterManager()
    
    def test_register_router(self, router_manager):
        """Test registering a router"""
        router = APIRouter(prefix="/test", tags=["test"])
        
        router_manager.register_router(
            router=router,
            name="test_router",
            prefix="/test",
            tags=["test"],
            description="Test router"
        )
        
        assert len(router_manager.routers) == 1
        assert "test_router" in router_manager.router_registry
    
    def test_get_all_routers(self, router_manager):
        """Test getting all routers"""
        router1 = APIRouter(prefix="/router1")
        router2 = APIRouter(prefix="/router2")
        
        router_manager.register_router(router1, "router1")
        router_manager.register_router(router2, "router2")
        
        routers = router_manager.get_all_routers()
        
        assert len(routers) == 2
        assert router1 in routers
        assert router2 in routers
    
    def test_get_router_info(self, router_manager):
        """Test getting router information"""
        router = APIRouter(prefix="/test", tags=["test"])
        
        router_manager.register_router(
            router=router,
            name="test_router",
            description="Test description"
        )
        
        info = router_manager.get_router_info()
        
        assert "test_router" in info
        assert info["test_router"]["prefix"] == "/test"
        assert info["test_router"]["tags"] == ["test"]
    
    def test_get_endpoints_summary(self, router_manager):
        """Test getting endpoints summary"""
        router = APIRouter(prefix="/test")
        
        @router.get("/endpoint1")
        def endpoint1():
            return "response1"
        
        @router.post("/endpoint2")
        def endpoint2():
            return "response2"
        
        router_manager.register_router(router, "test_router")
        
        summary = router_manager.get_endpoints_summary()
        
        assert "test_router" in summary
        assert len(summary["test_router"]["routes"]) >= 2


class TestGetRouterManager:
    """Tests for get_router_manager function"""
    
    def test_get_router_manager_singleton(self):
        """Test that get_router_manager returns singleton"""
        manager1 = get_router_manager()
        manager2 = get_router_manager()
        
        # Should return same instance (singleton pattern)
        assert manager1 is manager2


class TestServiceLocator:
    """Tests for ServiceLocator"""
    
    @pytest.fixture
    def service_locator(self):
        """Create service locator"""
        return ServiceLocator()
    
    def test_register_service(self, service_locator):
        """Test registering a service"""
        mock_service = Mock()
        
        service_locator.register_service("test_service", mock_service)
        
        assert "test_service" in service_locator._services
        assert service_locator._services["test_service"] == mock_service
    
    def test_get_service(self, service_locator):
        """Test getting a service"""
        mock_service = Mock()
        service_locator.register_service("test_service", mock_service)
        
        service = service_locator.get_service("test_service")
        
        assert service == mock_service
    
    def test_get_service_not_found(self, service_locator):
        """Test getting non-existent service"""
        with pytest.raises(ValueError, match="Service.*not registered"):
            service_locator.get_service("non_existent")
    
    def test_initialize_services(self, service_locator):
        """Test initializing services from dictionary"""
        services = {
            "service1": Mock(),
            "service2": Mock(),
            "service3": Mock()
        }
        
        service_locator.initialize_services(services)
        
        assert service_locator.is_initialized() is True
        assert len(service_locator._services) == 3
    
    def test_is_initialized(self, service_locator):
        """Test checking initialization status"""
        assert service_locator.is_initialized() is False
        
        service_locator.initialize_services({"test": Mock()})
        
        assert service_locator.is_initialized() is True


class TestGetServiceLocator:
    """Tests for get_service_locator function"""
    
    def test_get_service_locator_singleton(self):
        """Test that get_service_locator returns singleton"""
        locator1 = get_service_locator()
        locator2 = get_service_locator()
        
        # Should return same instance (singleton pattern)
        assert locator1 is locator2


class TestServiceRegistry:
    """Tests for ServiceRegistry"""
    
    @pytest.fixture
    def service_registry(self):
        """Create service registry"""
        from api.service_registry import ServiceRegistry
        return ServiceRegistry()
    
    def test_register_service(self, service_registry):
        """Test registering a service"""
        mock_service = Mock()
        
        service_registry.register("test_service", mock_service)
        
        service = service_registry.get("test_service")
        assert service == mock_service
    
    def test_get_all_services(self, service_registry):
        """Test getting all services"""
        service_registry.register("service1", Mock())
        service_registry.register("service2", Mock())
        
        all_services = service_registry.get_all()
        
        assert len(all_services) == 2
        assert "service1" in all_services
        assert "service2" in all_services


class TestFeatureFlagManager:
    """Tests for FeatureFlagManager"""
    
    @pytest.fixture
    def feature_flag_manager(self):
        """Create feature flag manager"""
        return FeatureFlagManager()
    
    def test_enable_feature(self, feature_flag_manager):
        """Test enabling a feature flag"""
        feature_flag_manager.enable("new_feature")
        
        assert feature_flag_manager.is_enabled("new_feature") is True
    
    def test_disable_feature(self, feature_flag_manager):
        """Test disabling a feature flag"""
        feature_flag_manager.enable("feature")
        feature_flag_manager.disable("feature")
        
        assert feature_flag_manager.is_enabled("feature") is False
    
    def test_is_enabled_default(self, feature_flag_manager):
        """Test checking feature flag that doesn't exist"""
        # Should return False for non-existent features
        assert feature_flag_manager.is_enabled("non_existent") is False
    
    def test_get_all_flags(self, feature_flag_manager):
        """Test getting all feature flags"""
        feature_flag_manager.enable("feature1")
        feature_flag_manager.enable("feature2")
        feature_flag_manager.disable("feature3")
        
        all_flags = feature_flag_manager.get_all()
        
        assert "feature1" in all_flags
        assert "feature2" in all_flags
        assert all_flags["feature1"] is True
        assert all_flags["feature2"] is True




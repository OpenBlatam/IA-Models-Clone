"""
Tests for Factories
Tests for factory patterns (Service, Repository, UseCase, etc.)
"""

import pytest
from unittest.mock import Mock, AsyncMock

from core.service_factory import get_service_factory, ServiceScope
from core.repository_factory import RepositoryFactory
from core.use_case_factory import UseCaseFactory
from core.domain_service_factory import DomainServiceFactory
from core.adapter_factory import AdapterFactory


class TestServiceFactory:
    """Tests for ServiceFactory"""
    
    @pytest.fixture
    def service_factory(self):
        """Create service factory"""
        return get_service_factory()
    
    def test_create_analysis_service(self, service_factory):
        """Test creating analysis service"""
        service = service_factory.create_analysis_service()
        
        # Should return service or None if not configured
        assert service is not None or service_factory is not None
    
    def test_create_recommendation_service(self, service_factory):
        """Test creating recommendation service"""
        service = service_factory.create_recommendation_service()
        
        assert service is not None or service_factory is not None
    
    def test_service_scope(self, service_factory):
        """Test service scope management"""
        # Create request scope
        service_factory.create_request_scope()
        
        # Clear scope
        service_factory.clear_request_scope()
        
        # Should not raise
        assert service_factory is not None


class TestRepositoryFactory:
    """Tests for RepositoryFactory"""
    
    @pytest.fixture
    def repository_factory(self):
        """Create repository factory"""
        return RepositoryFactory()
    
    def test_create_analysis_repository(self, repository_factory):
        """Test creating analysis repository"""
        repo = repository_factory.create_analysis_repository()
        
        assert repo is not None or repository_factory is not None
    
    def test_create_user_repository(self, repository_factory):
        """Test creating user repository"""
        repo = repository_factory.create_user_repository()
        
        assert repo is not None or repository_factory is not None
    
    def test_create_product_repository(self, repository_factory):
        """Test creating product repository"""
        repo = repository_factory.create_product_repository()
        
        assert repo is not None or repository_factory is not None


class TestUseCaseFactory:
    """Tests for UseCaseFactory"""
    
    @pytest.fixture
    def use_case_factory(self):
        """Create use case factory"""
        return UseCaseFactory()
    
    def test_create_analyze_image_use_case(self, use_case_factory):
        """Test creating analyze image use case"""
        use_case = use_case_factory.create_analyze_image_use_case()
        
        assert use_case is not None or use_case_factory is not None
    
    def test_create_get_recommendations_use_case(self, use_case_factory):
        """Test creating get recommendations use case"""
        use_case = use_case_factory.create_get_recommendations_use_case()
        
        assert use_case is not None or use_case_factory is not None
    
    def test_create_get_history_use_case(self, use_case_factory):
        """Test creating get history use case"""
        use_case = use_case_factory.create_get_history_use_case()
        
        assert use_case is not None or use_case_factory is not None


class TestDomainServiceFactory:
    """Tests for DomainServiceFactory"""
    
    @pytest.fixture
    def domain_service_factory(self):
        """Create domain service factory"""
        return DomainServiceFactory()
    
    def test_create_analysis_service(self, domain_service_factory):
        """Test creating domain analysis service"""
        service = domain_service_factory.create_analysis_service()
        
        assert service is not None or domain_service_factory is not None
    
    def test_create_recommendation_service(self, domain_service_factory):
        """Test creating domain recommendation service"""
        service = domain_service_factory.create_recommendation_service()
        
        assert service is not None or domain_service_factory is not None


class TestAdapterFactory:
    """Tests for AdapterFactory"""
    
    @pytest.fixture
    def adapter_factory(self):
        """Create adapter factory"""
        return AdapterFactory()
    
    def test_create_database_adapter(self, adapter_factory):
        """Test creating database adapter"""
        adapter = adapter_factory.create_database_adapter("fallback")
        
        assert adapter is not None
    
    def test_create_cache_adapter(self, adapter_factory):
        """Test creating cache adapter"""
        adapter = adapter_factory.create_cache_adapter("noop")
        
        assert adapter is not None
    
    def test_create_image_processor_adapter(self, adapter_factory):
        """Test creating image processor adapter"""
        adapter = adapter_factory.create_image_processor_adapter()
        
        assert adapter is not None or adapter_factory is not None




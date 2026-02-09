"""
Test Fixtures
=============

Test fixtures for common scenarios.
"""

import pytest
from typing import Dict, Any
from aws.modules.testing.mocks import MockRepository, MockCache, MockMessaging
from aws.modules.composition.service_composer import ServiceComposer
from aws.modules.business.service_factory import ServiceFactory


class TestFixtures:
    """Test fixtures."""
    
    @staticmethod
    @pytest.fixture
    def mock_repository():
        """Mock repository fixture."""
        return MockRepository()
    
    @staticmethod
    @pytest.fixture
    def mock_cache():
        """Mock cache fixture."""
        return MockCache()
    
    @staticmethod
    @pytest.fixture
    def mock_messaging():
        """Mock messaging fixture."""
        return MockMessaging()
    
    @staticmethod
    @pytest.fixture
    def service_composer(mock_repository, mock_cache, mock_messaging):
        """Service composer fixture with mocks."""
        return ServiceComposer(
            service_name="test-service",
            repository=mock_repository,
            cache=mock_cache,
            messaging=mock_messaging
        )
    
    @staticmethod
    @pytest.fixture
    def service_factory(mock_repository, mock_cache, mock_messaging):
        """Service factory fixture with mocks."""
        return ServiceFactory(
            repository=mock_repository,
            cache=mock_cache,
            messaging=mock_messaging
        )
















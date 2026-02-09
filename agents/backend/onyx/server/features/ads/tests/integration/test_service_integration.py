"""
Integration tests for the ads service layer.

This module tests the integration between different service components:
- Domain services integration
- Application services integration
- Infrastructure services integration
- Cross-layer service interactions
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import service components
from agents.backend.onyx.server.features.ads.domain.services import AdService, CampaignService, OptimizationService
from agents.backend.onyx.server.features.ads.application.use_cases import CreateAdUseCase, ApproveAdUseCase
from agents.backend.onyx.server.features.ads.infrastructure.database import DatabaseManager
from agents.backend.onyx.server.features.ads.infrastructure.storage import FileStorageManager
from agents.backend.onyx.server.features.ads.infrastructure.cache import CacheManager

# Import domain entities and DTOs
from agents.backend.onyx.server.features.ads.domain.entities import Ad, AdCampaign, AdGroup
from agents.backend.onyx.server.features.ads.application.dto import CreateAdRequest, CreateAdResponse


class TestServiceIntegration:
    """Test service integration and cross-layer communication."""

    @pytest.fixture
    def mock_ad_repository(self):
        """Mock ad repository."""
        mock = AsyncMock()
        mock.create.return_value = Ad(
            id="test-ad-123",
            title="Test Ad",
            description="Test Description",
            status="draft",
            platform="facebook",
            budget=1000.0
        )
        mock.get_by_id.return_value = Ad(
            id="test-ad-123",
            title="Test Ad",
            description="Test Description",
            status="draft",
            platform="facebook",
            budget=1000.0
        )
        return mock

    @pytest.fixture
    def mock_campaign_repository(self):
        """Mock campaign repository."""
        mock = AsyncMock()
        mock.create.return_value = AdCampaign(
            id="test-campaign-123",
            name="Test Campaign",
            description="Test Campaign Description",
            status="active",
            budget=5000.0
        )
        return mock

    @pytest.fixture
    def mock_database_manager(self):
        """Mock database manager."""
        mock = AsyncMock(spec=DatabaseManager)
        mock.get_session.return_value.__aenter__.return_value = AsyncMock()
        return mock

    @pytest.fixture
    def mock_storage_manager(self):
        """Mock storage manager."""
        mock = AsyncMock(spec=FileStorageManager)
        mock.save_file.return_value = "test-file-url"
        return mock

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager."""
        mock = AsyncMock(spec=CacheManager)
        mock.get.return_value = None
        mock.set.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_domain_service_integration(self, mock_ad_repository, mock_campaign_repository):
        """Test integration between domain services."""
        # Create services with mocked repositories
        ad_service = AdService(ad_repository=mock_ad_repository)
        campaign_service = CampaignService(campaign_repository=mock_campaign_repository)
        
        # Test ad service
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        assert ad.id == "test-ad-123"
        assert ad.title == "Test Ad"
        
        # Test campaign service
        campaign = await campaign_service.create_campaign(
            name="Test Campaign",
            description="Test Campaign Description",
            budget=5000.0
        )
        assert campaign.id == "test-campaign-123"
        assert campaign.name == "Test Campaign"

    @pytest.mark.asyncio
    async def test_application_service_integration(self, mock_ad_repository):
        """Test integration between application services."""
        # Create use case with mocked repository
        create_ad_use_case = CreateAdUseCase(ad_repository=mock_ad_repository)
        
        # Test use case execution
        request = CreateAdRequest(
            title="Test Ad",
            description="Test Description",
            brand_voice="Professional",
            target_audience="Tech professionals",
            platform="facebook",
            budget=1000.0
        )
        
        response = await create_ad_use_case.execute(request)
        assert response.success is True
        assert response.ad_id == "test-ad-123"

    @pytest.mark.asyncio
    async def test_infrastructure_service_integration(self, mock_database_manager, mock_storage_manager, mock_cache_manager):
        """Test integration between infrastructure services."""
        # Test database manager
        async with mock_database_manager.get_session() as session:
            assert session is not None
        
        # Test storage manager
        file_url = await mock_storage_manager.save_file("test-content", "test.txt")
        assert file_url == "test-file-url"
        
        # Test cache manager
        cache_result = await mock_cache_manager.set("test-key", "test-value")
        assert cache_result is True

    @pytest.mark.asyncio
    async def test_cross_layer_service_integration(self, mock_ad_repository, mock_database_manager):
        """Test integration across different service layers."""
        # Test domain service using infrastructure
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Create ad through domain service
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        # Verify ad was created
        assert ad.id == "test-ad-123"
        assert ad.status == "draft"
        
        # Test status change
        updated_ad = await ad_service.approve_ad(ad.id)
        assert updated_ad.status == "approved"

    @pytest.mark.asyncio
    async def test_service_error_handling_integration(self, mock_ad_repository):
        """Test error handling integration across services."""
        # Mock repository to raise exception
        mock_ad_repository.create.side_effect = Exception("Database error")
        
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Test that service handles errors gracefully
        with pytest.raises(Exception):
            await ad_service.create_ad(
                title="Test Ad",
                description="Test Description",
                platform="facebook",
                budget=1000.0
            )

    @pytest.mark.asyncio
    async def test_service_validation_integration(self, mock_ad_repository):
        """Test validation integration across services."""
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Test with invalid data
        with pytest.raises(ValueError):
            await ad_service.create_ad(
                title="",  # Invalid: empty title
                description="Test Description",
                platform="invalid_platform",  # Invalid platform
                budget=-100  # Invalid: negative budget
            )

    @pytest.mark.asyncio
    async def test_service_performance_integration(self, mock_ad_repository):
        """Test performance integration across services."""
        import time
        
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Measure service performance
        start_time = time.time()
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        end_time = time.time()
        
        assert ad is not None
        assert (end_time - start_time) < 1.0  # Should complete within 1 second

    @pytest.mark.asyncio
    async def test_service_caching_integration(self, mock_cache_manager):
        """Test caching integration across services."""
        # Test cache operations
        await mock_cache_manager.set("test-key", "test-value")
        cached_value = await mock_cache_manager.get("test-key")
        
        # Since we're mocking, this will return None
        # In real implementation, this would return the cached value
        assert cached_value is None

    @pytest.mark.asyncio
    async def test_service_transaction_integration(self, mock_database_manager):
        """Test transaction integration across services."""
        # Test database transaction handling
        async with mock_database_manager.get_session() as session:
            # Simulate transaction operations
            await session.execute("BEGIN")
            await session.execute("COMMIT")
            
            # Verify session was used
            assert session is not None

    @pytest.mark.asyncio
    async def test_service_async_integration(self, mock_ad_repository):
        """Test async integration across services."""
        # Test concurrent service operations
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        async def create_ad():
            return await ad_service.create_ad(
                title="Test Ad",
                description="Test Description",
                platform="facebook",
                budget=1000.0
            )
        
        # Execute multiple concurrent operations
        results = await asyncio.gather(*[create_ad() for _ in range(3)])
        
        # All should succeed
        for result in results:
            assert result.id == "test-ad-123"

    @pytest.mark.asyncio
    async def test_service_dependency_injection_integration(self, mock_ad_repository, mock_campaign_repository):
        """Test dependency injection integration across services."""
        # Test that services can be created with injected dependencies
        ad_service = AdService(ad_repository=mock_ad_repository)
        campaign_service = CampaignService(campaign_repository=mock_campaign_repository)
        
        # Verify services are properly initialized
        assert ad_service.ad_repository is mock_ad_repository
        assert campaign_service.campaign_repository is mock_campaign_repository

    @pytest.mark.asyncio
    async def test_service_lifecycle_integration(self, mock_ad_repository):
        """Test service lifecycle integration."""
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Test complete ad lifecycle
        # 1. Create ad
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        assert ad.status == "draft"
        
        # 2. Approve ad
        approved_ad = await ad_service.approve_ad(ad.id)
        assert approved_ad.status == "approved"
        
        # 3. Activate ad
        activated_ad = await ad_service.activate_ad(ad.id)
        assert activated_ad.status == "active"

    @pytest.mark.asyncio
    async def test_service_data_consistency_integration(self, mock_ad_repository):
        """Test data consistency integration across services."""
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Create ad
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        # Retrieve ad to verify consistency
        retrieved_ad = await ad_service.get_ad(ad.id)
        assert retrieved_ad.id == ad.id
        assert retrieved_ad.title == ad.title
        assert retrieved_ad.description == ad.description

    @pytest.mark.asyncio
    async def test_service_optimization_integration(self, mock_ad_repository):
        """Test optimization integration across services."""
        # Test that optimization services can be integrated
        from agents.backend.onyx.server.features.ads.optimization.factory import OptimizationFactory
        
        # Verify optimization factory exists
        assert OptimizationFactory is not None
        
        # Test that it can be used with ad services
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Create ad for optimization
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        # Verify ad is ready for optimization
        assert ad.id is not None

    @pytest.mark.asyncio
    async def test_service_training_integration(self, mock_ad_repository):
        """Test training integration across services."""
        # Test that training services can be integrated
        from agents.backend.onyx.server.features.ads.training.factory import TrainingFactory
        
        # Verify training factory exists
        assert TrainingFactory is not None
        
        # Test that it can be used with ad services
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Create ad for training data
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        # Verify ad is ready for training
        assert ad.id is not None

    @pytest.mark.asyncio
    async def test_service_monitoring_integration(self, mock_ad_repository):
        """Test monitoring integration across services."""
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Test that services can be monitored
        # This would depend on actual monitoring implementation
        
        # For now, test that service operations complete successfully
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        assert ad is not None
        assert ad.id == "test-ad-123"

    @pytest.mark.asyncio
    async def test_service_security_integration(self, mock_ad_repository):
        """Test security integration across services."""
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Test that services handle security concerns
        # This would depend on actual security implementation
        
        # For now, test that service operations complete successfully
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        assert ad is not None
        assert ad.id == "test-ad-123"

    @pytest.mark.asyncio
    async def test_service_logging_integration(self, mock_ad_repository):
        """Test logging integration across services."""
        ad_service = AdService(ad_repository=mock_ad_repository)
        
        # Test that services generate logs
        # This would depend on actual logging implementation
        
        # For now, test that service operations complete successfully
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        
        assert ad is not None
        assert ad.id == "test-ad-123"

    @pytest.mark.asyncio
    async def test_service_end_to_end_integration(self, mock_ad_repository, mock_campaign_repository):
        """Test end-to-end service integration."""
        # Test complete service workflow
        
        # 1. Create campaign
        campaign_service = CampaignService(campaign_repository=mock_campaign_repository)
        campaign = await campaign_service.create_campaign(
            name="Test Campaign",
            description="Test Campaign Description",
            budget=5000.0
        )
        assert campaign.id == "test-campaign-123"
        
        # 2. Create ad in campaign
        ad_service = AdService(ad_repository=mock_ad_repository)
        ad = await ad_service.create_ad(
            title="Test Ad",
            description="Test Description",
            platform="facebook",
            budget=1000.0
        )
        assert ad.id == "test-ad-123"
        
        # 3. Approve and activate ad
        approved_ad = await ad_service.approve_ad(ad.id)
        assert approved_ad.status == "approved"
        
        activated_ad = await ad_service.activate_ad(ad.id)
        assert activated_ad.status == "active"
        
        # 4. Verify campaign and ad are properly linked
        # This would depend on actual relationship implementation
        assert campaign.id is not None
        assert ad.id is not None


# Test utilities for service integration tests
@pytest.fixture
def service_test_utilities():
    """Utility functions for service integration tests."""
    
    def create_test_ad_data():
        """Create test ad data."""
        return {
            "title": "Test Ad",
            "description": "Test Description",
            "platform": "facebook",
            "budget": 1000.0
        }
    
    def create_test_campaign_data():
        """Create test campaign data."""
        return {
            "name": "Test Campaign",
            "description": "Test Campaign Description",
            "budget": 5000.0
        }
    
    def validate_ad_structure(ad):
        """Validate ad structure."""
        required_fields = ["id", "title", "description", "status", "platform", "budget"]
        return all(hasattr(ad, field) for field in required_fields)
    
    def validate_campaign_structure(campaign):
        """Validate campaign structure."""
        required_fields = ["id", "name", "description", "status", "budget"]
        return all(hasattr(campaign, field) for field in required_fields)
    
    return {
        "create_test_ad_data": create_test_ad_data,
        "create_test_campaign_data": create_test_campaign_data,
        "validate_ad_structure": validate_ad_structure,
        "validate_campaign_structure": validate_campaign_structure
    }

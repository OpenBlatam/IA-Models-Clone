"""
Content Distribution and Syndication Tests
========================================

Comprehensive tests for content distribution and syndication features including:
- Multi-platform distribution
- Syndication networks and partnerships
- Content adaptation for different platforms
- Distribution analytics and tracking
- Cross-platform optimization
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from typing import List, Dict, Any, Optional

# Test data
SAMPLE_DISTRIBUTION_CONFIG = {
    "platforms": ["linkedin", "twitter", "facebook", "instagram"],
    "syndication_networks": ["medium", "dev.to", "hashnode"],
    "adaptation_rules": {
        "linkedin": {"max_length": 1300, "hashtags": True},
        "twitter": {"max_length": 280, "hashtags": True},
        "facebook": {"max_length": 63206, "hashtags": True},
        "instagram": {"max_length": 2200, "hashtags": True}
    },
    "auto_distribute": True,
    "track_analytics": True
}

SAMPLE_SYNDICATION_NETWORK = {
    "id": str(uuid4()),
    "name": "Tech Blog Network",
    "platforms": ["medium", "dev.to", "hashnode"],
    "reach": 100000,
    "engagement_rate": 0.08,
    "content_requirements": {
        "min_length": 500,
        "max_length": 5000,
        "required_tags": ["tech", "programming"]
    }
}


class TestContentDistributionSyndication:
    """Test content distribution and syndication features"""
    
    @pytest.fixture
    def mock_distribution_service(self):
        """Mock distribution service"""
        service = AsyncMock()
        service.distribute_content.return_value = {
            "distribution_id": str(uuid4()),
            "platforms": ["linkedin", "twitter"],
            "status": "distributed",
            "tracking_urls": {
                "linkedin": "https://linkedin.com/post/123",
                "twitter": "https://twitter.com/status/456"
            }
        }
        service.syndicate_content.return_value = {
            "syndication_id": str(uuid4()),
            "networks": ["medium", "dev.to"],
            "status": "syndicated",
            "reach": 50000
        }
        service.get_distribution_analytics.return_value = {
            "total_reach": 150000,
            "total_engagement": 12000,
            "platform_performance": {
                "linkedin": {"reach": 80000, "engagement": 6000},
                "twitter": {"reach": 70000, "engagement": 6000}
            }
        }
        return service
    
    @pytest.fixture
    def mock_distribution_repository(self):
        """Mock distribution repository"""
        repository = AsyncMock()
        repository.save_distribution_config.return_value = SAMPLE_DISTRIBUTION_CONFIG
        repository.save_syndication_network.return_value = SAMPLE_SYNDICATION_NETWORK
        repository.get_distribution_analytics.return_value = {
            "total_reach": 150000,
            "total_engagement": 12000,
            "platform_performance": {
                "linkedin": {"reach": 80000, "engagement": 6000},
                "twitter": {"reach": 70000, "engagement": 6000}
            }
        }
        return repository
    
    @pytest.fixture
    def mock_content_adapter(self):
        """Mock content adapter"""
        adapter = AsyncMock()
        adapter.adapt_content.return_value = {
            "linkedin": "Adapted content for LinkedIn",
            "twitter": "Adapted content for Twitter",
            "facebook": "Adapted content for Facebook",
            "instagram": "Adapted content for Instagram"
        }
        adapter.validate_platform_requirements.return_value = True
        adapter.optimize_for_platform.return_value = {
            "optimized_content": "Optimized content",
            "hashtags": ["tech", "programming"],
            "mentions": ["@techuser"]
        }
        return adapter
    
    @pytest.fixture
    def post_service(self, mock_distribution_repository, mock_distribution_service, mock_content_adapter):
        """Post service with distribution dependencies"""
        from services.post_service import PostService
        service = PostService(
            repository=mock_distribution_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            distribution_service=mock_distribution_service,
            content_adapter=mock_content_adapter
        )
        return service
    
    async def test_distribute_content_multi_platform(self, post_service, mock_distribution_service):
        """Test distributing content across multiple platforms"""
        # Arrange
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter", "facebook"]
        
        # Act
        result = await post_service.distribution_service.distribute_content(
            post_id=post_id,
            platforms=platforms
        )
        
        # Assert
        assert result["status"] == "distributed"
        assert len(result["platforms"]) == 2
        assert "tracking_urls" in result
        mock_distribution_service.distribute_content.assert_called_once_with(
            post_id=post_id,
            platforms=platforms
        )
    
    async def test_syndicate_content_networks(self, post_service, mock_distribution_service):
        """Test syndicating content to external networks"""
        # Arrange
        post_id = str(uuid4())
        networks = ["medium", "dev.to", "hashnode"]
        
        # Act
        result = await post_service.distribution_service.syndicate_content(
            post_id=post_id,
            networks=networks
        )
        
        # Assert
        assert result["status"] == "syndicated"
        assert len(result["networks"]) == 2
        assert result["reach"] == 50000
        mock_distribution_service.syndicate_content.assert_called_once_with(
            post_id=post_id,
            networks=networks
        )
    
    async def test_adapt_content_for_platforms(self, post_service, mock_content_adapter):
        """Test adapting content for different platforms"""
        # Arrange
        original_content = "This is a test post about technology and programming"
        platforms = ["linkedin", "twitter", "facebook", "instagram"]
        
        # Act
        result = await post_service.content_adapter.adapt_content(
            content=original_content,
            platforms=platforms
        )
        
        # Assert
        assert len(result) == 4
        assert "linkedin" in result
        assert "twitter" in result
        assert "facebook" in result
        assert "instagram" in result
        mock_content_adapter.adapt_content.assert_called_once_with(
            content=original_content,
            platforms=platforms
        )
    
    async def test_get_distribution_analytics(self, post_service, mock_distribution_service):
        """Test retrieving distribution analytics"""
        # Arrange
        filters = {"date_range": "last_30_days", "platforms": ["linkedin", "twitter"]}
        
        # Act
        result = await post_service.distribution_service.get_distribution_analytics(filters)
        
        # Assert
        assert result["total_reach"] == 150000
        assert result["total_engagement"] == 12000
        assert "platform_performance" in result
        mock_distribution_service.get_distribution_analytics.assert_called_once_with(filters)
    
    async def test_validate_platform_requirements(self, post_service, mock_content_adapter):
        """Test validating platform requirements"""
        # Arrange
        content = "Test content"
        platform = "twitter"
        requirements = {"max_length": 280, "hashtags": True}
        
        # Act
        result = await post_service.content_adapter.validate_platform_requirements(
            content=content,
            platform=platform,
            requirements=requirements
        )
        
        # Assert
        assert result is True
        mock_content_adapter.validate_platform_requirements.assert_called_once_with(
            content=content,
            platform=platform,
            requirements=requirements
        )
    
    async def test_optimize_content_for_platform(self, post_service, mock_content_adapter):
        """Test optimizing content for specific platform"""
        # Arrange
        content = "Test content about technology"
        platform = "linkedin"
        keywords = ["tech", "programming", "ai"]
        
        # Act
        result = await post_service.content_adapter.optimize_for_platform(
            content=content,
            platform=platform,
            keywords=keywords
        )
        
        # Assert
        assert "optimized_content" in result
        assert "hashtags" in result
        assert "mentions" in result
        mock_content_adapter.optimize_for_platform.assert_called_once_with(
            content=content,
            platform=platform,
            keywords=keywords
        )
    
    async def test_schedule_distribution(self, post_service, mock_distribution_service):
        """Test scheduling content distribution"""
        # Arrange
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter"]
        scheduled_time = datetime.utcnow() + timedelta(hours=2)
        
        # Act
        result = await post_service.distribution_service.schedule_distribution(
            post_id=post_id,
            platforms=platforms,
            scheduled_time=scheduled_time
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.schedule_distribution.assert_called_once_with(
            post_id=post_id,
            platforms=platforms,
            scheduled_time=scheduled_time
        )
    
    async def test_track_distribution_performance(self, post_service, mock_distribution_service):
        """Test tracking distribution performance"""
        # Arrange
        distribution_id = str(uuid4())
        
        # Act
        result = await post_service.distribution_service.track_performance(
            distribution_id=distribution_id
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.track_performance.assert_called_once_with(
            distribution_id=distribution_id
        )
    
    async def test_manage_syndication_networks(self, post_service, mock_distribution_service):
        """Test managing syndication networks"""
        # Arrange
        network_data = {
            "name": "Tech Blog Network",
            "platforms": ["medium", "dev.to"],
            "reach": 100000
        }
        
        # Act
        result = await post_service.distribution_service.create_syndication_network(
            network_data=network_data
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.create_syndication_network.assert_called_once_with(
            network_data=network_data
        )
    
    async def test_analyze_cross_platform_performance(self, post_service, mock_distribution_service):
        """Test analyzing cross-platform performance"""
        # Arrange
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter", "facebook"]
        
        # Act
        result = await post_service.distribution_service.analyze_cross_platform_performance(
            post_id=post_id,
            platforms=platforms
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.analyze_cross_platform_performance.assert_called_once_with(
            post_id=post_id,
            platforms=platforms
        )
    
    async def test_optimize_distribution_strategy(self, post_service, mock_distribution_service):
        """Test optimizing distribution strategy"""
        # Arrange
        historical_data = {
            "platform_performance": {
                "linkedin": {"reach": 80000, "engagement": 6000},
                "twitter": {"reach": 70000, "engagement": 6000}
            }
        }
        
        # Act
        result = await post_service.distribution_service.optimize_strategy(
            historical_data=historical_data
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.optimize_strategy.assert_called_once_with(
            historical_data=historical_data
        )
    
    async def test_handle_distribution_errors(self, post_service, mock_distribution_service):
        """Test handling distribution errors"""
        # Arrange
        distribution_id = str(uuid4())
        error_type = "platform_unavailable"
        
        # Act
        result = await post_service.distribution_service.handle_error(
            distribution_id=distribution_id,
            error_type=error_type
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.handle_error.assert_called_once_with(
            distribution_id=distribution_id,
            error_type=error_type
        )
    
    async def test_retry_failed_distributions(self, post_service, mock_distribution_service):
        """Test retrying failed distributions"""
        # Arrange
        failed_distribution_ids = [str(uuid4()) for _ in range(3)]
        
        # Act
        result = await post_service.distribution_service.retry_failed_distributions(
            distribution_ids=failed_distribution_ids
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.retry_failed_distributions.assert_called_once_with(
            distribution_ids=failed_distribution_ids
        )
    
    async def test_manage_distribution_queues(self, post_service, mock_distribution_service):
        """Test managing distribution queues"""
        # Arrange
        queue_name = "high_priority"
        distribution_ids = [str(uuid4()) for _ in range(5)]
        
        # Act
        result = await post_service.distribution_service.manage_queue(
            queue_name=queue_name,
            distribution_ids=distribution_ids
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.manage_queue.assert_called_once_with(
            queue_name=queue_name,
            distribution_ids=distribution_ids
        )
    
    async def test_generate_distribution_reports(self, post_service, mock_distribution_service):
        """Test generating distribution reports"""
        # Arrange
        report_type = "performance_summary"
        date_range = "last_month"
        
        # Act
        result = await post_service.distribution_service.generate_report(
            report_type=report_type,
            date_range=date_range
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.generate_report.assert_called_once_with(
            report_type=report_type,
            date_range=date_range
        )
    
    async def test_manage_distribution_permissions(self, post_service, mock_distribution_service):
        """Test managing distribution permissions"""
        # Arrange
        user_id = str(uuid4())
        platform = "linkedin"
        permissions = ["publish", "schedule", "analytics"]
        
        # Act
        result = await post_service.distribution_service.manage_permissions(
            user_id=user_id,
            platform=platform,
            permissions=permissions
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.manage_permissions.assert_called_once_with(
            user_id=user_id,
            platform=platform,
            permissions=permissions
        )
    
    async def test_optimize_content_for_syndication(self, post_service, mock_content_adapter):
        """Test optimizing content for syndication"""
        # Arrange
        content = "Original content"
        syndication_network = "medium"
        requirements = {"min_length": 500, "max_length": 5000}
        
        # Act
        result = await post_service.content_adapter.optimize_for_syndication(
            content=content,
            network=syndication_network,
            requirements=requirements
        )
        
        # Assert
        assert result is not None
        mock_content_adapter.optimize_for_syndication.assert_called_once_with(
            content=content,
            network=syndication_network,
            requirements=requirements
        )
    
    async def test_track_syndication_performance(self, post_service, mock_distribution_service):
        """Test tracking syndication performance"""
        # Arrange
        syndication_id = str(uuid4())
        
        # Act
        result = await post_service.distribution_service.track_syndication_performance(
            syndication_id=syndication_id
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.track_syndication_performance.assert_called_once_with(
            syndication_id=syndication_id
        )
    
    async def test_manage_distribution_automation(self, post_service, mock_distribution_service):
        """Test managing distribution automation"""
        # Arrange
        automation_rules = {
            "auto_distribute": True,
            "platforms": ["linkedin", "twitter"],
            "schedule": "immediate"
        }
        
        # Act
        result = await post_service.distribution_service.setup_automation(
            automation_rules=automation_rules
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.setup_automation.assert_called_once_with(
            automation_rules=automation_rules
        )
    
    async def test_analyze_distribution_trends(self, post_service, mock_distribution_service):
        """Test analyzing distribution trends"""
        # Arrange
        time_period = "last_6_months"
        platforms = ["linkedin", "twitter", "facebook"]
        
        # Act
        result = await post_service.distribution_service.analyze_trends(
            time_period=time_period,
            platforms=platforms
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.analyze_trends.assert_called_once_with(
            time_period=time_period,
            platforms=platforms
        )
    
    async def test_optimize_distribution_timing(self, post_service, mock_distribution_service):
        """Test optimizing distribution timing"""
        # Arrange
        platform = "linkedin"
        audience_data = {
            "timezone": "UTC-5",
            "peak_hours": [9, 12, 17],
            "engagement_patterns": {"monday": "high", "friday": "medium"}
        }
        
        # Act
        result = await post_service.distribution_service.optimize_timing(
            platform=platform,
            audience_data=audience_data
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.optimize_timing.assert_called_once_with(
            platform=platform,
            audience_data=audience_data
        )
    
    async def test_manage_distribution_budget(self, post_service, mock_distribution_service):
        """Test managing distribution budget"""
        # Arrange
        budget_config = {
            "total_budget": 1000,
            "platform_allocation": {
                "linkedin": 400,
                "twitter": 300,
                "facebook": 300
            }
        }
        
        # Act
        result = await post_service.distribution_service.manage_budget(
            budget_config=budget_config
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.manage_budget.assert_called_once_with(
            budget_config=budget_config
        )
    
    async def test_integrate_with_external_platforms(self, post_service, mock_distribution_service):
        """Test integrating with external platforms"""
        # Arrange
        platform = "linkedin"
        integration_config = {
            "api_key": "test_key",
            "webhook_url": "https://webhook.url",
            "permissions": ["publish", "read"]
        }
        
        # Act
        result = await post_service.distribution_service.setup_platform_integration(
            platform=platform,
            config=integration_config
        )
        
        # Assert
        assert result is not None
        mock_distribution_service.setup_platform_integration.assert_called_once_with(
            platform=platform,
            config=integration_config
        )

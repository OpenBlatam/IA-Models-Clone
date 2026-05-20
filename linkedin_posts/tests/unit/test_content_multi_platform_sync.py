"""
Content Multi-Platform Synchronization Tests
==========================================

Comprehensive tests for content multi-platform synchronization features including:
- Cross-platform content management
- Synchronization strategies and conflict resolution
- Platform-specific content adaptations
- Multi-platform analytics and reporting
- Cross-platform engagement tracking
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from typing import List, Dict, Any, Optional

# Test data
SAMPLE_SYNC_CONFIG = {
    "platforms": ["linkedin", "twitter", "facebook", "instagram"],
    "sync_strategies": {
        "linkedin": "primary",
        "twitter": "cross_post",
        "facebook": "adapted",
        "instagram": "visual_adapted"
    },
    "conflict_resolution": {
        "strategy": "latest_wins",
        "merge_enabled": True,
        "auto_resolve": True
    },
    "adaptation_rules": {
        "content_length": {
            "linkedin": "unlimited",
            "twitter": "280",
            "facebook": "unlimited",
            "instagram": "2200"
        },
        "hashtag_limit": {
            "linkedin": "unlimited",
            "twitter": "unlimited",
            "facebook": "unlimited",
            "instagram": "30"
        }
    }
}

SAMPLE_PLATFORM_DATA = {
    "linkedin": {
        "post_id": "li_123456",
        "engagement": {"likes": 150, "comments": 25, "shares": 10},
        "reach": 5000,
        "impressions": 8000
    },
    "twitter": {
        "post_id": "tw_789012",
        "engagement": {"likes": 200, "retweets": 45, "replies": 15},
        "reach": 3000,
        "impressions": 6000
    },
    "facebook": {
        "post_id": "fb_345678",
        "engagement": {"likes": 300, "comments": 50, "shares": 20},
        "reach": 8000,
        "impressions": 12000
    }
}

class TestContentMultiPlatformSync:
    """Test content multi-platform synchronization features"""
    
    @pytest.fixture
    def mock_sync_service(self):
        """Mock synchronization service."""
        service = AsyncMock()
        service.sync_to_platforms.return_value = {
            "linkedin": {"success": True, "post_id": "li_123456"},
            "twitter": {"success": True, "post_id": "tw_789012"},
            "facebook": {"success": True, "post_id": "fb_345678"}
        }
        service.get_sync_status.return_value = {
            "linkedin": "synced",
            "twitter": "synced",
            "facebook": "synced"
        }
        service.resolve_conflicts.return_value = {
            "resolved": True,
            "merged_content": "Updated content with platform adaptations"
        }
        return service
    
    @pytest.fixture
    def mock_platform_service(self):
        """Mock platform-specific service."""
        service = AsyncMock()
        service.adapt_content_for_platform.return_value = {
            "adapted_content": "Platform-specific content",
            "adaptations_applied": ["length", "hashtags", "format"]
        }
        service.get_platform_analytics.return_value = SAMPLE_PLATFORM_DATA
        service.validate_platform_content.return_value = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        return service
    
    @pytest.fixture
    def mock_sync_repository(self):
        """Mock synchronization repository."""
        repository = AsyncMock()
        repository.save_sync_data.return_value = {
            "sync_id": str(uuid4()),
            "timestamp": datetime.now(),
            "status": "completed"
        }
        repository.get_sync_history.return_value = [
            {
                "sync_id": str(uuid4()),
                "platforms": ["linkedin", "twitter"],
                "status": "completed",
                "timestamp": datetime.now() - timedelta(hours=1)
            }
        ]
        repository.get_platform_data.return_value = SAMPLE_PLATFORM_DATA
        return repository
    
    @pytest.fixture
    def post_service(self, mock_sync_repository, mock_sync_service, mock_platform_service):
        from services.post_service import PostService
        service = PostService(
            repository=mock_sync_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            sync_service=mock_sync_service,
            platform_service=mock_platform_service
        )
        return service
    
    @pytest.mark.asyncio
    async def test_multi_platform_sync_creation(self, post_service, mock_sync_service):
        """Test creating multi-platform synchronization."""
        post_data = {
            "content": "Test post for multiple platforms",
            "platforms": ["linkedin", "twitter", "facebook"],
            "sync_strategy": "cross_post"
        }
        
        result = await post_service.create_multi_platform_sync(post_data)
        
        assert result["success"] is True
        assert "sync_id" in result
        assert len(result["platform_results"]) == 3
        mock_sync_service.sync_to_platforms.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_content_adaptation(self, post_service, mock_platform_service):
        """Test adapting content for different platforms."""
        content = "Original content with #hashtags and @mentions"
        platforms = ["linkedin", "twitter", "instagram"]
        
        adaptations = await post_service.adapt_content_for_platforms(content, platforms)
        
        assert len(adaptations) == 3
        for platform, adaptation in adaptations.items():
            assert "adapted_content" in adaptation
            assert "adaptations_applied" in adaptation
        mock_platform_service.adapt_content_for_platform.assert_called()
    
    @pytest.mark.asyncio
    async def test_sync_conflict_resolution(self, post_service, mock_sync_service):
        """Test resolving synchronization conflicts."""
        conflicts = {
            "linkedin": {"content": "Updated on LinkedIn"},
            "twitter": {"content": "Updated on Twitter"},
            "facebook": {"content": "Updated on Facebook"}
        }
        
        resolution = await post_service.resolve_sync_conflicts(conflicts)
        
        assert resolution["resolved"] is True
        assert "merged_content" in resolution
        mock_sync_service.resolve_conflicts.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_platform_analytics_aggregation(self, post_service, mock_platform_service):
        """Test aggregating analytics from multiple platforms."""
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter", "facebook"]
        
        analytics = await post_service.get_multi_platform_analytics(post_id, platforms)
        
        assert "total_engagement" in analytics
        assert "total_reach" in analytics
        assert "platform_breakdown" in analytics
        assert len(analytics["platform_breakdown"]) == 3
        mock_platform_service.get_platform_analytics.assert_called()
    
    @pytest.mark.asyncio
    async def test_sync_status_monitoring(self, post_service, mock_sync_service):
        """Test monitoring synchronization status across platforms."""
        sync_id = str(uuid4())
        
        status = await post_service.get_sync_status(sync_id)
        
        assert "linkedin" in status
        assert "twitter" in status
        assert "facebook" in status
        mock_sync_service.get_sync_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_content_validation(self, post_service, mock_platform_service):
        """Test validating content for different platforms."""
        content = "Test content with #hashtags"
        platforms = ["linkedin", "twitter", "instagram"]
        
        validations = await post_service.validate_platform_content(content, platforms)
        
        assert len(validations) == 3
        for platform, validation in validations.items():
            assert "valid" in validation
            assert "warnings" in validation
            assert "errors" in validation
        mock_platform_service.validate_platform_content.assert_called()
    
    @pytest.mark.asyncio
    async def test_sync_history_retrieval(self, post_service, mock_sync_repository):
        """Test retrieving synchronization history."""
        post_id = str(uuid4())
        
        history = await post_service.get_sync_history(post_id)
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert "sync_id" in history[0]
        assert "platforms" in history[0]
        assert "status" in history[0]
        mock_sync_repository.get_sync_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_data_persistence(self, post_service, mock_sync_repository):
        """Test persisting platform-specific data."""
        sync_data = {
            "post_id": str(uuid4()),
            "platforms": ["linkedin", "twitter"],
            "sync_results": SAMPLE_PLATFORM_DATA
        }
        
        result = await post_service.save_sync_data(sync_data)
        
        assert "sync_id" in result
        assert "timestamp" in result
        assert "status" in result
        mock_sync_repository.save_sync_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cross_platform_engagement_tracking(self, post_service, mock_platform_service):
        """Test tracking engagement across multiple platforms."""
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter", "facebook"]
        
        engagement = await post_service.track_cross_platform_engagement(post_id, platforms)
        
        assert "total_engagement" in engagement
        assert "platform_engagement" in engagement
        assert "engagement_trends" in engagement
        mock_platform_service.get_platform_analytics.assert_called()
    
    @pytest.mark.asyncio
    async def test_sync_error_handling(self, post_service, mock_sync_service):
        """Test handling synchronization errors."""
        mock_sync_service.sync_to_platforms.side_effect = Exception("Sync failed")
        
        post_data = {
            "content": "Test post",
            "platforms": ["linkedin", "twitter"]
        }
        
        with pytest.raises(Exception):
            await post_service.create_multi_platform_sync(post_data)
    
    @pytest.mark.asyncio
    async def test_platform_specific_optimization(self, post_service, mock_platform_service):
        """Test optimizing content for specific platforms."""
        content = "Original content"
        platform = "twitter"
        
        optimization = await post_service.optimize_for_platform(content, platform)
        
        assert "optimized_content" in optimization
        assert "optimizations_applied" in optimization
        mock_platform_service.adapt_content_for_platform.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_schedule_management(self, post_service, mock_sync_repository):
        """Test managing synchronization schedules."""
        schedule_data = {
            "post_id": str(uuid4()),
            "platforms": ["linkedin", "twitter"],
            "schedule_time": datetime.now() + timedelta(hours=1),
            "sync_strategy": "scheduled"
        }
        
        result = await post_service.create_sync_schedule(schedule_data)
        
        assert "schedule_id" in result
        assert "status" in result
        mock_sync_repository.save_sync_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_content_metrics(self, post_service, mock_platform_service):
        """Test calculating platform-specific content metrics."""
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter", "facebook"]
        
        metrics = await post_service.get_platform_content_metrics(post_id, platforms)
        
        assert "content_performance" in metrics
        assert "engagement_rates" in metrics
        assert "reach_metrics" in metrics
        mock_platform_service.get_platform_analytics.assert_called()
    
    @pytest.mark.asyncio
    async def test_sync_automation_rules(self, post_service, mock_sync_service):
        """Test applying automation rules for synchronization."""
        automation_rules = {
            "auto_sync": True,
            "conflict_resolution": "auto",
            "platform_priorities": ["linkedin", "twitter", "facebook"]
        }
        
        result = await post_service.apply_sync_automation(automation_rules)
        
        assert result["automation_applied"] is True
        assert "rules_applied" in result
        mock_sync_service.sync_to_platforms.assert_called()
    
    @pytest.mark.asyncio
    async def test_platform_content_archiving(self, post_service, mock_sync_repository):
        """Test archiving platform-specific content."""
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter"]
        
        result = await post_service.archive_platform_content(post_id, platforms)
        
        assert result["archived"] is True
        assert "archived_platforms" in result
        mock_sync_repository.save_sync_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_performance_monitoring(self, post_service, mock_sync_service):
        """Test monitoring synchronization performance."""
        sync_id = str(uuid4())
        
        performance = await post_service.monitor_sync_performance(sync_id)
        
        assert "sync_duration" in performance
        assert "platform_performance" in performance
        assert "error_rate" in performance
        mock_sync_service.get_sync_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_content_backup(self, post_service, mock_sync_repository):
        """Test backing up platform-specific content."""
        post_id = str(uuid4())
        platforms = ["linkedin", "twitter", "facebook"]
        
        backup = await post_service.backup_platform_content(post_id, platforms)
        
        assert backup["backup_created"] is True
        assert "backup_id" in backup
        assert "backup_platforms" in backup
        mock_sync_repository.save_sync_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_data_export(self, post_service, mock_sync_repository):
        """Test exporting synchronization data."""
        post_id = str(uuid4())
        export_format = "json"
        
        export = await post_service.export_sync_data(post_id, export_format)
        
        assert "export_data" in export
        assert "export_format" in export
        assert "export_timestamp" in export
        mock_sync_repository.get_sync_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_content_restoration(self, post_service, mock_sync_repository):
        """Test restoring platform-specific content."""
        backup_id = str(uuid4())
        platforms = ["linkedin", "twitter"]
        
        restoration = await post_service.restore_platform_content(backup_id, platforms)
        
        assert restoration["restored"] is True
        assert "restored_platforms" in restoration
        mock_sync_repository.save_sync_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_analytics_reporting(self, post_service, mock_platform_service):
        """Test generating synchronization analytics reports."""
        time_range = {
            "start": datetime.now() - timedelta(days=7),
            "end": datetime.now()
        }
        platforms = ["linkedin", "twitter", "facebook"]
        
        report = await post_service.generate_sync_analytics_report(time_range, platforms)
        
        assert "sync_metrics" in report
        assert "platform_performance" in report
        assert "engagement_analysis" in report
        mock_platform_service.get_platform_analytics.assert_called()
    
    @pytest.mark.asyncio
    async def test_platform_content_validation_rules(self, post_service, mock_platform_service):
        """Test applying validation rules for platform content."""
        content = "Test content"
        platform = "twitter"
        validation_rules = {
            "max_length": 280,
            "hashtag_limit": 30,
            "mention_limit": 10
        }
        
        validation = await post_service.validate_with_rules(content, platform, validation_rules)
        
        assert "valid" in validation
        assert "rule_violations" in validation
        assert "suggestions" in validation
        mock_platform_service.validate_platform_content.assert_called_once()

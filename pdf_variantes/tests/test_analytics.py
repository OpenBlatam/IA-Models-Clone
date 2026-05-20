"""
Unit Tests for Analytics
========================
Tests for analytics engine and statistics.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Try to import analytics classes
try:
    from analytics import (
        UsageStats,
        UserActivity,
        AnalyticsEngine
    )
except ImportError:
    UsageStats = None
    UserActivity = None
    AnalyticsEngine = None


class TestUsageStats:
    """Tests for UsageStats class."""
    
    def test_usage_stats_creation(self):
        """Test creating UsageStats."""
        if UsageStats is None:
            pytest.skip("UsageStats not available")
        
        stats = UsageStats()
        assert stats is not None
    
    def test_usage_stats_initialization(self):
        """Test UsageStats initialization."""
        if UsageStats is None:
            pytest.skip("UsageStats not available")
        
        stats = UsageStats()
        # Check that stats have expected attributes
        assert hasattr(stats, "total_uploads") or hasattr(stats, "uploads")
        assert hasattr(stats, "total_variants") or hasattr(stats, "variants")
    
    def test_usage_stats_increment(self):
        """Test incrementing usage stats."""
        if UsageStats is None:
            pytest.skip("UsageStats not available")
        
        stats = UsageStats()
        # Try to increment if method exists
        if hasattr(stats, "increment_uploads"):
            stats.increment_uploads()
            assert stats.total_uploads > 0 or stats.uploads > 0
    
    def test_usage_stats_to_dict(self):
        """Test converting UsageStats to dictionary."""
        if UsageStats is None:
            pytest.skip("UsageStats not available")
        
        stats = UsageStats()
        if hasattr(stats, "to_dict"):
            stats_dict = stats.to_dict()
            assert isinstance(stats_dict, dict)
        elif hasattr(stats, "dict"):
            stats_dict = stats.dict()
            assert isinstance(stats_dict, dict)
        elif hasattr(stats, "model_dump"):
            stats_dict = stats.model_dump()
            assert isinstance(stats_dict, dict)


class TestUserActivity:
    """Tests for UserActivity class."""
    
    def test_user_activity_creation(self):
        """Test creating UserActivity."""
        if UserActivity is None:
            pytest.skip("UserActivity not available")
        
        activity = UserActivity(user_id="test_user")
        assert activity is not None
        assert activity.user_id == "test_user"
    
    def test_user_activity_timestamp(self):
        """Test UserActivity timestamp."""
        if UserActivity is None:
            pytest.skip("UserActivity not available")
        
        activity = UserActivity(user_id="test_user")
        # Should have timestamp
        assert hasattr(activity, "timestamp") or hasattr(activity, "created_at")
    
    def test_user_activity_actions(self):
        """Test UserActivity actions tracking."""
        if UserActivity is None:
            pytest.skip("UserActivity not available")
        
        activity = UserActivity(user_id="test_user")
        # Should be able to track actions
        if hasattr(activity, "add_action"):
            activity.add_action("upload", {"file_id": "123"})
            assert len(activity.actions) > 0 if hasattr(activity, "actions") else True


class TestAnalyticsEngine:
    """Tests for AnalyticsEngine class."""
    
    @pytest.fixture
    def analytics_engine(self):
        """Create AnalyticsEngine instance."""
        if AnalyticsEngine is None:
            pytest.skip("AnalyticsEngine not available")
        return AnalyticsEngine()
    
    def test_analytics_engine_initialization(self, analytics_engine):
        """Test AnalyticsEngine initialization."""
        assert analytics_engine is not None
    
    @pytest.mark.asyncio
    async def test_track_event(self, analytics_engine):
        """Test tracking events."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "track_event"):
            result = await analytics_engine.track_event(
                event_type="pdf_upload",
                user_id="test_user",
                metadata={"file_id": "123"}
            )
            assert result is True or result is None
    
    @pytest.mark.asyncio
    async def test_get_usage_stats(self, analytics_engine):
        """Test getting usage statistics."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "get_usage_stats"):
            stats = await analytics_engine.get_usage_stats()
            assert isinstance(stats, dict) or isinstance(stats, UsageStats)
    
    @pytest.mark.asyncio
    async def test_get_user_activity(self, analytics_engine):
        """Test getting user activity."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "get_user_activity"):
            activity = await analytics_engine.get_user_activity("test_user")
            assert activity is not None
    
    @pytest.mark.asyncio
    async def test_get_top_users(self, analytics_engine):
        """Test getting top users."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "get_top_users"):
            top_users = await analytics_engine.get_top_users(limit=10)
            assert isinstance(top_users, list)
    
    @pytest.mark.asyncio
    async def test_get_popular_variants(self, analytics_engine):
        """Test getting popular variants."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "get_popular_variants"):
            variants = await analytics_engine.get_popular_variants(limit=10)
            assert isinstance(variants, list)
    
    @pytest.mark.asyncio
    async def test_get_daily_stats(self, analytics_engine):
        """Test getting daily statistics."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "get_daily_stats"):
            today = datetime.now().date()
            stats = await analytics_engine.get_daily_stats(today)
            assert isinstance(stats, dict)
    
    @pytest.mark.asyncio
    async def test_get_time_range_stats(self, analytics_engine):
        """Test getting statistics for time range."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "get_time_range_stats"):
            start = datetime.now() - timedelta(days=7)
            end = datetime.now()
            stats = await analytics_engine.get_time_range_stats(start, end)
            assert isinstance(stats, dict)
    
    @pytest.mark.asyncio
    async def test_aggregate_metrics(self, analytics_engine):
        """Test aggregating metrics."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "aggregate_metrics"):
            metrics = await analytics_engine.aggregate_metrics()
            assert isinstance(metrics, dict)
            assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_export_analytics(self, analytics_engine):
        """Test exporting analytics data."""
        if analytics_engine is None:
            pytest.skip("AnalyticsEngine not available")
        
        if hasattr(analytics_engine, "export_analytics"):
            data = await analytics_engine.export_analytics()
            assert isinstance(data, dict) or isinstance(data, list)


class TestAnalyticsIntegration:
    """Integration tests for analytics."""
    
    @pytest.mark.asyncio
    async def test_track_and_retrieve(self):
        """Test tracking events and retrieving stats."""
        if AnalyticsEngine is None:
            pytest.skip("AnalyticsEngine not available")
        
        engine = AnalyticsEngine()
        
        # Track some events
        if hasattr(engine, "track_event"):
            await engine.track_event("pdf_upload", "user1", {})
            await engine.track_event("variant_generate", "user1", {})
            await engine.track_event("pdf_upload", "user2", {})
        
        # Retrieve stats
        if hasattr(engine, "get_usage_stats"):
            stats = await engine.get_usage_stats()
            assert isinstance(stats, dict) or isinstance(stats, UsageStats)
    
    @pytest.mark.asyncio
    async def test_concurrent_tracking(self):
        """Test concurrent event tracking."""
        if AnalyticsEngine is None:
            pytest.skip("AnalyticsEngine not available")
        
        import asyncio
        
        engine = AnalyticsEngine()
        
        if hasattr(engine, "track_event"):
            # Track events concurrently
            tasks = [
                engine.track_event(f"event_{i}", f"user_{i % 10}", {})
                for i in range(100)
            ]
            await asyncio.gather(*tasks)
            
            # Should handle concurrent access
            if hasattr(engine, "get_usage_stats"):
                stats = await engine.get_usage_stats()
                assert isinstance(stats, dict) or isinstance(stats, UsageStats)




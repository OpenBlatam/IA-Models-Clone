"""
Test repository fixtures for the ads feature.

This module provides comprehensive test repository fixtures for:
- Domain repositories (AdRepository, CampaignRepository, etc.)
- Infrastructure repositories (concrete implementations)
- Mock data access patterns
- Test scenario repositories
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

# Import repository interfaces and implementations
from agents.backend.onyx.server.features.ads.domain.repositories import (
    AdRepository, CampaignRepository, GroupRepository, PerformanceRepository,
    AnalyticsRepository, OptimizationRepository
)
from agents.backend.onyx.server.features.ads.infrastructure.repositories import (
    AdsRepositoryImpl, CampaignRepositoryImpl, GroupRepositoryImpl,
    PerformanceRepositoryImpl, AnalyticsRepositoryImpl, OptimizationRepositoryImpl,
    RepositoryFactory
)


class TestRepositoryFixtures:
    """Test repository fixtures for ads feature testing."""

    @pytest.fixture
    def mock_ad_repository(self):
        """Mock ad repository for testing."""
        mock = AsyncMock(spec=AdRepository)
        
        # Mock ad data
        mock_ad = Mock()
        mock_ad.id = "test-ad-123"
        mock_ad.title = "Test Advertisement"
        mock_ad.description = "A test advertisement for testing purposes"
        mock_ad.status = "draft"
        mock_ad.platform = "facebook"
        mock_ad.budget = 1000.0
        mock_ad.created_at = datetime.now()
        mock_ad.updated_at = datetime.now()
        
        # Mock CRUD operations
        mock.create.return_value = mock_ad
        mock.get_by_id.return_value = mock_ad
        mock.update.return_value = mock_ad
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_ad, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_ad]
        mock.find_by_status.return_value = [mock_ad]
        mock.find_by_platform.return_value = [mock_ad]
        mock.find_by_budget_range.return_value = [mock_ad]
        
        # Mock batch operations
        mock.create_many.return_value = [mock_ad, Mock(), Mock()]
        mock.update_many.return_value = 3
        mock.delete_many.return_value = 3
        
        return mock

    @pytest.fixture
    def mock_campaign_repository(self):
        """Mock campaign repository for testing."""
        mock = AsyncMock(spec=CampaignRepository)
        
        # Mock campaign data
        mock_campaign = Mock()
        mock_campaign.id = "test-campaign-123"
        mock_campaign.name = "Test Campaign"
        mock_campaign.description = "A test advertising campaign"
        mock_campaign.status = "active"
        mock_campaign.budget = 5000.0
        mock_campaign.start_date = datetime.now()
        mock_campaign.end_date = datetime.now() + timedelta(days=30)
        mock_campaign.created_at = datetime.now()
        mock_campaign.updated_at = datetime.now()
        
        # Mock CRUD operations
        mock.create.return_value = mock_campaign
        mock.get_by_id.return_value = mock_campaign
        mock.update.return_value = mock_campaign
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_campaign, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_campaign]
        mock.find_by_status.return_value = [mock_campaign]
        mock.find_by_budget_range.return_value = [mock_campaign]
        mock.find_active_campaigns.return_value = [mock_campaign]
        
        # Mock relationship operations
        mock.get_campaign_ads.return_value = [Mock(), Mock()]
        mock.get_campaign_performance.return_value = Mock()
        
        return mock

    @pytest.fixture
    def mock_group_repository(self):
        """Mock ad group repository for testing."""
        mock = AsyncMock(spec=GroupRepository)
        
        # Mock group data
        mock_group = Mock()
        mock_group.id = "test-group-123"
        mock_group.name = "Test Ad Group"
        mock_group.description = "A test ad group within a campaign"
        mock_group.status = "active"
        mock_group.campaign_id = "test-campaign-123"
        mock_group.budget = 2000.0
        mock_group.created_at = datetime.now()
        mock_group.updated_at = datetime.now()
        
        # Mock CRUD operations
        mock.create.return_value = mock_group
        mock.get_by_id.return_value = mock_group
        mock.update.return_value = mock_group
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_group, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_group]
        mock.find_by_campaign.return_value = [mock_group]
        mock.find_by_status.return_value = [mock_group]
        
        # Mock relationship operations
        mock.get_group_ads.return_value = [Mock(), Mock()]
        mock.get_group_performance.return_value = Mock()
        
        return mock

    @pytest.fixture
    def mock_performance_repository(self):
        """Mock performance repository for testing."""
        mock = AsyncMock(spec=PerformanceRepository)
        
        # Mock performance data
        mock_performance = Mock()
        mock_performance.id = "test-performance-123"
        mock_performance.ad_id = "test-ad-123"
        mock_performance.date = datetime.now().date()
        mock_performance.impressions = 1000
        mock_performance.clicks = 50
        mock_performance.conversions = 5
        mock_performance.spend = 150.0
        mock_performance.ctr = 0.05
        mock_performance.cpc = 3.0
        mock_performance.cpm = 150.0
        mock_performance.conversion_rate = 0.10
        mock_performance.roas = 2.5
        
        # Mock CRUD operations
        mock.create.return_value = mock_performance
        mock.get_by_id.return_value = mock_performance
        mock.update.return_value = mock_performance
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_performance, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_performance]
        mock.find_by_ad.return_value = [mock_performance]
        mock.find_by_date_range.return_value = [mock_performance]
        mock.find_by_metrics.return_value = [mock_performance]
        
        # Mock aggregation operations
        mock.get_performance_summary.return_value = {
            "total_impressions": 5000,
            "total_clicks": 250,
            "total_conversions": 25,
            "total_spend": 750.0,
            "average_ctr": 0.05,
            "average_cpc": 3.0
        }
        
        mock.get_performance_trends.return_value = [
            {"date": "2024-01-01", "ctr": 0.04, "conversion_rate": 0.08},
            {"date": "2024-01-02", "ctr": 0.05, "conversion_rate": 0.10}
        ]
        
        return mock

    @pytest.fixture
    def mock_analytics_repository(self):
        """Mock analytics repository for testing."""
        mock = AsyncMock(spec=AnalyticsRepository)
        
        # Mock analytics data
        mock_analytics = Mock()
        mock_analytics.id = "test-analytics-123"
        mock_analytics.report_type = "performance_summary"
        mock_analytics.data = {
            "total_ads": 100,
            "total_campaigns": 25,
            "total_spend": 50000.0,
            "overall_roas": 2.8
        }
        mock_analytics.generated_at = datetime.now()
        
        # Mock CRUD operations
        mock.create.return_value = mock_analytics
        mock.get_by_id.return_value = mock_analytics
        mock.update.return_value = mock_analytics
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_analytics, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_analytics]
        mock.find_by_type.return_value = [mock_analytics]
        mock.find_by_date_range.return_value = [mock_analytics]
        
        # Mock analytics operations
        mock.generate_report.return_value = mock_analytics
        mock.get_report_types.return_value = ["performance_summary", "audience_insights", "trend_analysis"]
        mock.export_report.return_value = {"export_id": "test-export-123"}
        
        return mock

    @pytest.fixture
    def mock_optimization_repository(self):
        """Mock optimization repository for testing."""
        mock = AsyncMock(spec=OptimizationRepository)
        
        # Mock optimization data
        mock_optimization = Mock()
        mock_optimization.id = "test-optimization-123"
        mock_optimization.ad_id = "test-ad-123"
        mock_optimization.optimization_type = "performance"
        mock_optimization.status = "completed"
        mock_optimization.improvements = {"ctr": 0.15, "conversion_rate": 0.12}
        mock_optimization.recommendations = ["Increase bid by 15%", "Refine targeting"]
        mock_optimization.created_at = datetime.now()
        mock_optimization.completed_at = datetime.now()
        
        # Mock CRUD operations
        mock.create.return_value = mock_optimization
        mock.get_by_id.return_value = mock_optimization
        mock.update.return_value = mock_optimization
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_optimization, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_optimization]
        mock.find_by_ad.return_value = [mock_optimization]
        mock.find_by_type.return_value = [mock_optimization]
        mock.find_by_status.return_value = [mock_optimization]
        
        # Mock optimization operations
        mock.get_optimization_history.return_value = [mock_optimization, Mock()]
        mock.get_optimization_stats.return_value = {
            "total_optimizations": 50,
            "successful_optimizations": 45,
            "average_improvement": 0.15
        }
        
        return mock

    @pytest.fixture
    def mock_ads_repository_impl(self):
        """Mock ads repository implementation for testing."""
        mock = AsyncMock(spec=AdsRepositoryImpl)
        
        # Mock database manager
        mock.database_manager = AsyncMock()
        mock.database_manager.get_session.return_value.__aenter__.return_value = AsyncMock()
        
        # Mock CRUD operations
        mock_ad = Mock()
        mock_ad.id = "test-ad-123"
        mock_ad.title = "Test Advertisement"
        mock_ad.status = "draft"
        
        mock.create.return_value = mock_ad
        mock.get_by_id.return_value = mock_ad
        mock.update.return_value = mock_ad
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_ad, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_ad]
        mock.find_by_status.return_value = [mock_ad]
        mock.find_by_platform.return_value = [mock_ad]
        
        return mock

    @pytest.fixture
    def mock_campaign_repository_impl(self):
        """Mock campaign repository implementation for testing."""
        mock = AsyncMock(spec=CampaignRepositoryImpl)
        
        # Mock database manager
        mock.database_manager = AsyncMock()
        mock.database_manager.get_session.return_value.__aenter__.return_value = AsyncMock()
        
        # Mock CRUD operations
        mock_campaign = Mock()
        mock_campaign.id = "test-campaign-123"
        mock_campaign.name = "Test Campaign"
        mock_campaign.status = "active"
        
        mock.create.return_value = mock_campaign
        mock.get_by_id.return_value = mock_campaign
        mock.update.return_value = mock_campaign
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_campaign, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_campaign]
        mock.find_by_status.return_value = [mock_campaign]
        mock.find_active_campaigns.return_value = [mock_campaign]
        
        return mock

    @pytest.fixture
    def mock_group_repository_impl(self):
        """Mock group repository implementation for testing."""
        mock = AsyncMock(spec=GroupRepositoryImpl)
        
        # Mock database manager
        mock.database_manager = AsyncMock()
        mock.database_manager.get_session.return_value.__aenter__.return_value = AsyncMock()
        
        # Mock CRUD operations
        mock_group = Mock()
        mock_group.id = "test-group-123"
        mock_group.name = "Test Ad Group"
        mock_group.status = "active"
        
        mock.create.return_value = mock_group
        mock.get_by_id.return_value = mock_group
        mock.update.return_value = mock_group
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_group, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_group]
        mock.find_by_campaign.return_value = [mock_group]
        mock.find_by_status.return_value = [mock_group]
        
        return mock

    @pytest.fixture
    def mock_performance_repository_impl(self):
        """Mock performance repository implementation for testing."""
        mock = AsyncMock(spec=PerformanceRepositoryImpl)
        
        # Mock database manager
        mock.database_manager = AsyncMock()
        mock.database_manager.get_session.return_value.__aenter__.return_value = AsyncMock()
        
        # Mock CRUD operations
        mock_performance = Mock()
        mock_performance.id = "test-performance-123"
        mock_performance.ad_id = "test-ad-123"
        mock_performance.ctr = 0.05
        
        mock.create.return_value = mock_performance
        mock.get_by_id.return_value = mock_performance
        mock.update.return_value = mock_performance
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_performance, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_performance]
        mock.find_by_ad.return_value = [mock_performance]
        mock.find_by_date_range.return_value = [mock_performance]
        
        return mock

    @pytest.fixture
    def mock_analytics_repository_impl(self):
        """Mock analytics repository implementation for testing."""
        mock = AsyncMock(spec=AnalyticsRepositoryImpl)
        
        # Mock database manager
        mock.database_manager = AsyncMock()
        mock.database_manager.get_session.return_value.__aenter__.return_value = AsyncMock()
        
        # Mock CRUD operations
        mock_analytics = Mock()
        mock_analytics.id = "test-analytics-123"
        mock_analytics.report_type = "performance_summary"
        
        mock.create.return_value = mock_analytics
        mock.get_by_id.return_value = mock_analytics
        mock.update.return_value = mock_analytics
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_analytics, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_analytics]
        mock.find_by_type.return_value = [mock_analytics]
        mock.find_by_date_range.return_value = [mock_analytics]
        
        return mock

    @pytest.fixture
    def mock_optimization_repository_impl(self):
        """Mock optimization repository implementation for testing."""
        mock = AsyncMock(spec=OptimizationRepositoryImpl)
        
        # Mock database manager
        mock.database_manager = AsyncMock()
        mock.database_manager.get_session.return_value.__aenter__.return_value = AsyncMock()
        
        # Mock CRUD operations
        mock_optimization = Mock()
        mock_optimization.id = "test-optimization-123"
        mock_optimization.ad_id = "test-ad-123"
        mock_optimization.optimization_type = "performance"
        
        mock.create.return_value = mock_optimization
        mock.get_by_id.return_value = mock_optimization
        mock.update.return_value = mock_optimization
        mock.delete.return_value = True
        
        # Mock query operations
        mock.list.return_value = [mock_optimization, Mock(), Mock()]
        mock.count.return_value = 3
        mock.exists.return_value = True
        
        # Mock search operations
        mock.search.return_value = [mock_optimization]
        mock.find_by_ad.return_value = [mock_optimization]
        mock.find_by_type.return_value = [mock_optimization]
        
        return mock

    @pytest.fixture
    def mock_repository_factory(self):
        """Mock repository factory for testing."""
        mock = Mock(spec=RepositoryFactory)
        
        # Mock repository creation methods
        mock.create_ads_repository.return_value = AsyncMock(spec=AdsRepositoryImpl)
        mock.create_campaign_repository.return_value = AsyncMock(spec=CampaignRepositoryImpl)
        mock.create_group_repository.return_value = AsyncMock(spec=GroupRepositoryImpl)
        mock.create_performance_repository.return_value = AsyncMock(spec=PerformanceRepositoryImpl)
        mock.create_analytics_repository.return_value = AsyncMock(spec=AnalyticsRepositoryImpl)
        mock.create_optimization_repository.return_value = AsyncMock(spec=OptimizationRepositoryImpl)
        
        # Mock factory configuration
        mock.database_manager = AsyncMock()
        mock.config = {"pool_size": 5, "max_overflow": 10}
        
        return mock

    @pytest.fixture
    def mock_database_session(self):
        """Mock database session for testing."""
        mock = AsyncMock()
        
        # Mock session methods
        mock.execute.return_value = AsyncMock()
        mock.commit.return_value = None
        mock.rollback.return_value = None
        mock.close.return_value = None
        
        # Mock query results
        mock_result = AsyncMock()
        mock_result.scalars.return_value = [Mock(), Mock()]
        mock_result.first.return_value = Mock()
        mock_result.all.return_value = [Mock(), Mock()]
        mock_result.count.return_value = 2
        
        mock.execute.return_value = mock_result
        
        return mock

    @pytest.fixture
    def mock_database_connection(self):
        """Mock database connection for testing."""
        mock = AsyncMock()
        
        # Mock connection methods
        mock.execute.return_value = AsyncMock()
        mock.fetchone.return_value = {"id": 1, "name": "test"}
        mock.fetchall.return_value = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        mock.close.return_value = None
        
        # Mock query results
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_result.fetchone.return_value = {"id": 1, "name": "test"}
        mock_result.fetchall.return_value = [{"id": 1, "name": "test"}]
        
        mock.execute.return_value = mock_result
        
        return mock

    @pytest.fixture
    def mock_transaction_context(self):
        """Mock transaction context for testing."""
        mock = AsyncMock()
        
        # Mock context manager methods
        mock.__aenter__.return_value = mock
        mock.__aexit__.return_value = None
        
        # Mock transaction methods
        mock.begin.return_value = None
        mock.commit.return_value = None
        mock.rollback.return_value = None
        
        # Mock session access
        mock.session = AsyncMock()
        mock.session.execute.return_value = AsyncMock()
        
        return mock

    @pytest.fixture
    def mock_query_builder(self):
        """Mock query builder for testing."""
        mock = Mock()
        
        # Mock query building methods
        mock.select.return_value = mock
        mock.from_.return_value = mock
        mock.where.return_value = mock
        mock.order_by.return_value = mock
        mock.limit.return_value = mock
        mock.offset.return_value = mock
        mock.join.return_value = mock
        mock.group_by.return_value = mock
        mock.having.return_value = mock
        
        # Mock query execution
        mock.execute.return_value = [Mock(), Mock()]
        mock.first.return_value = Mock()
        mock.count.return_value = 2
        
        return mock

    @pytest.fixture
    def mock_data_mapper(self):
        """Mock data mapper for testing."""
        mock = Mock()
        
        # Mock mapping methods
        mock.to_entity.return_value = Mock()
        mock.to_dto.return_value = Mock()
        mock.to_database_model.return_value = Mock()
        
        # Mock batch mapping
        mock.map_to_entities.return_value = [Mock(), Mock()]
        mock.map_to_dtos.return_value = [Mock(), Mock()]
        mock.map_to_database_models.return_value = [Mock(), Mock()]
        
        # Mock validation
        mock.validate_entity.return_value = {"is_valid": True, "errors": []}
        mock.validate_dto.return_value = {"is_valid": True, "errors": []}
        
        return mock

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing."""
        mock = AsyncMock()
        
        # Mock cache operations
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        mock.exists.return_value = False
        mock.expire.return_value = True
        
        # Mock cache patterns
        mock.get_by_pattern.return_value = ["key1", "key2"]
        mock.delete_by_pattern.return_value = 2
        mock.clear.return_value = True
        
        # Mock cache statistics
        mock.get_stats.return_value = {
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.83
        }
        
        return mock

    @pytest.fixture
    def mock_connection_pool(self):
        """Mock connection pool for testing."""
        mock = AsyncMock()
        
        # Mock connection management
        mock.get_connection.return_value = AsyncMock()
        mock.return_connection.return_value = None
        mock.close_connection.return_value = None
        
        # Mock pool statistics
        mock.get_pool_stats.return_value = {
            "pool_size": 10,
            "active_connections": 3,
            "idle_connections": 7,
            "waiting_requests": 0
        }
        
        # Mock pool health
        mock.check_pool_health.return_value = {
            "status": "healthy",
            "response_time": 5
        }
        
        return mock

    @pytest.fixture
    def mock_query_executor(self):
        """Mock query executor for testing."""
        mock = AsyncMock()
        
        # Mock query execution
        mock.execute_query.return_value = [Mock(), Mock()]
        mock.execute_scalar.return_value = 42
        mock.execute_non_query.return_value = 1
        
        # Mock query building
        mock.build_query.return_value = "SELECT * FROM ads"
        mock.build_count_query.return_value = "SELECT COUNT(*) FROM ads"
        mock.build_search_query.return_value = "SELECT * FROM ads WHERE title LIKE %s"
        
        # Mock query optimization
        mock.optimize_query.return_value = "SELECT * FROM ads USE INDEX (idx_status)"
        mock.explain_query.return_value = {"type": "SIMPLE", "rows": 100}
        
        return mock

    @pytest.fixture
    def mock_result_processor(self):
        """Mock result processor for testing."""
        mock = Mock()
        
        # Mock result processing
        mock.process_single_result.return_value = Mock()
        mock.process_multiple_results.return_value = [Mock(), Mock()]
        mock.process_aggregation_result.return_value = {"count": 2, "sum": 100}
        
        # Mock result transformation
        mock.transform_to_entity.return_value = Mock()
        mock.transform_to_dto.return_value = Mock()
        mock.transform_to_dict.return_value = {"id": 1, "name": "test"}
        
        # Mock result validation
        mock.validate_result.return_value = {"is_valid": True, "errors": []}
        mock.sanitize_result.return_value = {"id": 1, "name": "sanitized_test"}
        
        return mock

    @pytest.fixture
    def mock_error_handler(self):
        """Mock error handler for testing."""
        mock = AsyncMock()
        
        # Mock error handling
        mock.handle_database_error.return_value = {"error_handled": True, "recovery_action": "retry"}
        mock.handle_connection_error.return_value = {"error_handled": True, "recovery_action": "reconnect"}
        mock.handle_query_error.return_value = {"error_handled": True, "recovery_action": "fallback"}
        
        # Mock error logging
        mock.log_error.return_value = None
        mock.log_error_with_context.return_value = None
        
        # Mock error recovery
        mock.attempt_recovery.return_value = {
            "recovery_attempted": True,
            "success": True,
            "actions_taken": ["retry", "fallback"]
        }
        
        return mock

    @pytest.fixture
    def mock_performance_monitor(self):
        """Mock performance monitor for testing."""
        mock = AsyncMock()
        
        # Mock performance tracking
        mock.track_query_performance.return_value = None
        mock.track_connection_performance.return_value = None
        mock.track_repository_performance.return_value = None
        
        # Mock performance metrics
        mock.get_query_metrics.return_value = {
            "total_queries": 1000,
            "average_query_time": 15.5,
            "slow_queries": 5
        }
        
        mock.get_connection_metrics.return_value = {
            "total_connections": 100,
            "average_connection_time": 5.2,
            "connection_errors": 2
        }
        
        # Mock performance alerts
        mock.check_performance_thresholds.return_value = {
            "thresholds_exceeded": False,
            "alerts": []
        }
        
        return mock

    @pytest.fixture
    def mock_data_validator(self):
        """Mock data validator for testing."""
        mock = Mock()
        
        # Mock validation methods
        mock.validate_entity_data.return_value = {"is_valid": True, "errors": []}
        mock.validate_dto_data.return_value = {"is_valid": True, "errors": []}
        mock.validate_query_parameters.return_value = {"is_valid": True, "errors": []}
        
        # Mock business rule validation
        mock.validate_business_rules.return_value = {
            "rules_passed": 5,
            "rules_failed": 0,
            "violations": []
        }
        
        # Mock constraint validation
        mock.validate_constraints.return_value = {
            "constraints_satisfied": True,
            "violations": []
        }
        
        return mock

    @pytest.fixture
    def mock_data_transformer(self):
        """Mock data transformer for testing."""
        mock = Mock()
        
        # Mock transformation methods
        mock.transform_to_entity.return_value = Mock()
        mock.transform_to_dto.return_value = Mock()
        mock.transform_to_database_model.return_value = Mock()
        
        # Mock batch transformation
        mock.transform_batch_to_entities.return_value = [Mock(), Mock()]
        mock.transform_batch_to_dtos.return_value = [Mock(), Mock()]
        mock.transform_batch_to_database_models.return_value = [Mock(), Mock()]
        
        # Mock field mapping
        mock.map_fields.return_value = {"id": "entity_id", "name": "entity_name"}
        mock.unmap_fields.return_value = {"entity_id": "id", "entity_name": "name"}
        
        return mock

    @pytest.fixture
    def mock_query_optimizer(self):
        """Mock query optimizer for testing."""
        mock = Mock()
        
        # Mock optimization methods
        mock.optimize_select_query.return_value = "SELECT id, name FROM ads USE INDEX (idx_status)"
        mock.optimize_search_query.return_value = "SELECT * FROM ads WHERE MATCH(title, description) AGAINST(%s)"
        mock.optimize_join_query.return_value = "SELECT a.*, c.name FROM ads a JOIN campaigns c ON a.campaign_id = c.id"
        
        # Mock index recommendations
        mock.recommend_indexes.return_value = [
            {"table": "ads", "columns": ["status", "platform"], "type": "BTREE"},
            {"table": "ads", "columns": ["title"], "type": "FULLTEXT"}
        ]
        
        # Mock query analysis
        mock.analyze_query_plan.return_value = {
            "type": "SIMPLE",
            "rows": 100,
            "cost": 10.5,
            "recommendations": ["Add index on status column"]
        }
        
        return mock

    @pytest.fixture
    def mock_batch_processor(self):
        """Mock batch processor for testing."""
        mock = AsyncMock()
        
        # Mock batch operations
        mock.process_batch_create.return_value = [Mock(), Mock(), Mock()]
        mock.process_batch_update.return_value = 3
        mock.process_batch_delete.return_value = 3
        
        # Mock batch validation
        mock.validate_batch_data.return_value = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Mock batch optimization
        mock.optimize_batch_size.return_value = 100
        mock.split_large_batch.return_value = [[Mock(), Mock()], [Mock()]]
        
        # Mock batch monitoring
        mock.monitor_batch_progress.return_value = {
            "total_items": 100,
            "processed_items": 75,
            "progress": 0.75
        }
        
        return mock

    @pytest.fixture
    def mock_audit_logger(self):
        """Mock audit logger for testing."""
        mock = AsyncMock()
        
        # Mock audit logging
        mock.log_repository_operation.return_value = {"log_id": "test-log-123"}
        mock.log_data_access.return_value = {"log_id": "test-access-log-123"}
        mock.log_query_execution.return_value = {"log_id": "test-query-log-123"}
        
        # Mock audit trail
        mock.get_audit_trail.return_value = [
            {"operation": "create", "table": "ads", "user": "test-user", "timestamp": datetime.now()},
            {"operation": "update", "table": "ads", "user": "test-user", "timestamp": datetime.now()}
        ]
        
        # Mock audit export
        mock.export_audit_log.return_value = {"export_id": "test-audit-export-123"}
        
        return mock


# Repository factory for creating mock repositories
@pytest.fixture
def mock_repository_factory():
    """Factory for creating mock repositories with custom configurations."""
    
    def create_mock_ad_repository(**kwargs):
        """Create a mock ad repository with custom configurations."""
        mock = AsyncMock(spec=AdRepository)
        
        # Set default values
        mock.create.return_value = kwargs.get("create_result", Mock())
        mock.get_by_id.return_value = kwargs.get("get_result", Mock())
        mock.update.return_value = kwargs.get("update_result", Mock())
        mock.delete.return_value = kwargs.get("delete_result", True)
        mock.list.return_value = kwargs.get("list_result", [Mock(), Mock(), Mock()])
        mock.count.return_value = kwargs.get("count_result", 3)
        
        return mock
    
    def create_mock_campaign_repository(**kwargs):
        """Create a mock campaign repository with custom configurations."""
        mock = AsyncMock(spec=CampaignRepository)
        
        # Set default values
        mock.create.return_value = kwargs.get("create_result", Mock())
        mock.get_by_id.return_value = kwargs.get("get_result", Mock())
        mock.update.return_value = kwargs.get("update_result", Mock())
        mock.delete.return_value = kwargs.get("delete_result", True)
        mock.list.return_value = kwargs.get("list_result", [Mock(), Mock(), Mock()])
        mock.count.return_value = kwargs.get("count_result", 3)
        
        return mock
    
    def create_mock_repository_impl(**kwargs):
        """Create a mock repository implementation with custom configurations."""
        mock = AsyncMock()
        
        # Set default values
        mock.create.return_value = kwargs.get("create_result", Mock())
        mock.get_by_id.return_value = kwargs.get("get_result", Mock())
        mock.update.return_value = kwargs.get("update_result", Mock())
        mock.delete.return_value = kwargs.get("delete_result", True)
        mock.list.return_value = kwargs.get("list_result", [Mock(), Mock(), Mock()])
        mock.count.return_value = kwargs.get("count_result", 3)
        
        # Mock database manager
        mock.database_manager = kwargs.get("database_manager", AsyncMock())
        
        return mock
    
    return {
        "create_mock_ad_repository": create_mock_ad_repository,
        "create_mock_campaign_repository": create_mock_campaign_repository,
        "create_mock_repository_impl": create_mock_repository_impl
    }


# Repository scenario fixtures
@pytest.fixture
def repository_scenario_fixtures():
    """Fixtures for different repository testing scenarios."""
    
    def get_happy_path_repositories():
        """Get repositories configured for happy path testing."""
        return {
            "ad_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="happy-ad-123"))),
            "campaign_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="happy-campaign-123"))),
            "performance_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="happy-performance-123")))
        }
    
    def get_error_scenario_repositories():
        """Get repositories configured for error scenario testing."""
        return {
            "ad_repository": AsyncMock(create=AsyncMock(side_effect=Exception("Database error"))),
            "campaign_repository": AsyncMock(create=AsyncMock(side_effect=ValueError("Invalid data"))),
            "performance_repository": AsyncMock(create=AsyncMock(side_effect=RuntimeError("Service unavailable")))
        }
    
    def get_performance_test_repositories():
        """Get repositories configured for performance testing."""
        return {
            "ad_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="perf-ad-123"))),
            "campaign_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="perf-campaign-123"))),
            "cache_manager": AsyncMock(get=AsyncMock(return_value="cached_data"))
        }
    
    def get_concurrent_test_repositories():
        """Get repositories configured for concurrent testing."""
        return {
            "ad_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="concurrent-ad-123"))),
            "campaign_repository": AsyncMock(create=AsyncMock(return_value=Mock(id="concurrent-campaign-123"))),
            "connection_pool": AsyncMock(get_connection=AsyncMock(return_value=AsyncMock()))
        }
    
    return {
        "get_happy_path_repositories": get_happy_path_repositories,
        "get_error_scenario_repositories": get_error_scenario_repositories,
        "get_performance_test_repositories": get_performance_test_repositories,
        "get_concurrent_test_repositories": get_concurrent_test_repositories
    }

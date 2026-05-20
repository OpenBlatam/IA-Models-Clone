"""
Test model fixtures for the ads feature.

This module provides comprehensive test models for:
- Database models and ORM objects
- API request/response models
- Mock objects and stubs
- Test scenario models
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

# Import domain entities and DTOs
from agents.backend.onyx.server.features.ads.domain.entities import Ad, AdCampaign, AdGroup, AdPerformance
from agents.backend.onyx.server.features.ads.domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria
from agents.backend.onyx.server.features.ads.application.dto import (
    CreateAdRequest, CreateAdResponse, ApproveAdRequest, ApproveAdResponse,
    ActivateAdRequest, ActivateAdResponse, PauseAdRequest, PauseAdResponse,
    CreateCampaignRequest, CreateCampaignResponse, OptimizationRequest, OptimizationResponse
)


class TestModelFixtures:
    """Test model fixtures for ads feature testing."""

    @pytest.fixture
    def mock_ad_model(self):
        """Mock ad model for testing."""
        mock = Mock(spec=Ad)
        mock.id = "test-ad-123"
        mock.title = "Test Advertisement"
        mock.description = "A test advertisement for testing purposes"
        mock.status = "draft"
        mock.ad_type = "display"
        mock.platform = "facebook"
        mock.budget = 1000.0
        mock.targeting_criteria = {
            "age_range": [25, 45],
            "interests": ["technology", "business"],
            "location": "United States",
            "gender": "all"
        }
        mock.creative_assets = {
            "images": ["image1.jpg", "image2.jpg"],
            "videos": ["video1.mp4"],
            "text": "Get the best deals on tech products!"
        }
        mock.metrics = {
            "impressions": 0,
            "clicks": 0,
            "conversions": 0,
            "spend": 0.0
        }
        mock.created_at = datetime.now()
        mock.updated_at = datetime.now()
        
        # Mock methods
        mock.approve.return_value = mock
        mock.activate.return_value = mock
        mock.pause.return_value = mock
        mock.archive.return_value = mock
        mock.update_metrics.return_value = mock
        
        return mock

    @pytest.fixture
    def mock_campaign_model(self):
        """Mock campaign model for testing."""
        mock = Mock(spec=AdCampaign)
        mock.id = "test-campaign-123"
        mock.name = "Test Campaign"
        mock.description = "A test advertising campaign"
        mock.status = "active"
        mock.budget = 5000.0
        mock.start_date = datetime.now()
        mock.end_date = datetime.now() + timedelta(days=30)
        mock.targeting_criteria = {
            "age_range": [18, 65],
            "interests": ["technology", "business", "finance"],
            "location": "United States",
            "gender": "all"
        }
        mock.created_at = datetime.now()
        mock.updated_at = datetime.now()
        
        # Mock methods
        mock.activate.return_value = mock
        mock.pause.return_value = mock
        mock.archive.return_value = mock
        mock.update_budget.return_value = mock
        
        return mock

    @pytest.fixture
    def mock_ad_group_model(self):
        """Mock ad group model for testing."""
        mock = Mock(spec=AdGroup)
        mock.id = "test-group-123"
        mock.name = "Test Ad Group"
        mock.description = "A test ad group within a campaign"
        mock.status = "active"
        mock.campaign_id = "test-campaign-123"
        mock.budget = 2000.0
        mock.targeting_criteria = {
            "age_range": [25, 45],
            "interests": ["technology"],
            "location": "California",
            "gender": "all"
        }
        mock.created_at = datetime.now()
        mock.updated_at = datetime.now()
        
        # Mock methods
        mock.activate.return_value = mock
        mock.pause.return_value = mock
        mock.archive.return_value = mock
        
        return mock

    @pytest.fixture
    def mock_performance_model(self):
        """Mock performance model for testing."""
        mock = Mock(spec=AdPerformance)
        mock.id = "test-performance-123"
        mock.ad_id = "test-ad-123"
        mock.date = datetime.now().date()
        mock.impressions = 1000
        mock.clicks = 50
        mock.conversions = 5
        mock.spend = 150.0
        mock.ctr = 0.05
        mock.cpc = 3.0
        mock.cpm = 150.0
        mock.conversion_rate = 0.10
        mock.roas = 2.5
        
        # Mock methods
        mock.update_metrics.return_value = mock
        mock.calculate_roas.return_value = 2.5
        
        return mock

    @pytest.fixture
    def mock_budget_model(self):
        """Mock budget model for testing."""
        mock = Mock(spec=Budget)
        mock.amount = 1000.0
        mock.currency = "USD"
        mock.duration_days = 30
        mock.daily_limit = 50.0
        mock.lifetime_limit = 1000.0
        mock.billing_type = "daily"
        
        # Mock methods
        mock.is_within_limits.return_value = True
        mock.get_remaining_budget.return_value = 500.0
        mock.can_spend.return_value = True
        
        return mock

    @pytest.fixture
    def mock_targeting_criteria_model(self):
        """Mock targeting criteria model for testing."""
        mock = Mock(spec=TargetingCriteria)
        mock.age_range = [25, 45]
        mock.interests = ["technology", "business", "finance"]
        mock.location = "United States"
        mock.gender = "all"
        mock.languages = ["English"]
        mock.devices = ["mobile", "desktop"]
        mock.platforms = ["facebook", "instagram"]
        mock.custom_audiences = ["high_value_customers"]
        mock.excluded_audiences = ["existing_customers"]
        
        # Mock methods
        mock.is_valid.return_value = True
        mock.get_coverage_estimate.return_value = 1000000
        mock.has_conflicts.return_value = False
        
        return mock

    @pytest.fixture
    def mock_create_ad_request_model(self):
        """Mock create ad request model for testing."""
        mock = Mock(spec=CreateAdRequest)
        mock.title = "Test Advertisement"
        mock.description = "A test advertisement for testing purposes"
        mock.brand_voice = "Professional and friendly"
        mock.target_audience = "Tech professionals aged 25-45"
        mock.platform = "facebook"
        mock.budget = 1000.0
        mock.targeting_criteria = {
            "age_range": [25, 45],
            "interests": ["technology", "business"],
            "location": "United States",
            "gender": "all"
        }
        mock.creative_assets = {
            "images": ["image1.jpg", "image2.jpg"],
            "videos": ["video1.mp4"],
            "text": "Get the best deals on tech products!"
        }
        
        # Mock methods
        mock.validate.return_value = True
        mock.to_dict.return_value = {
            "title": mock.title,
            "description": mock.description,
            "platform": mock.platform,
            "budget": mock.budget
        }
        
        return mock

    @pytest.fixture
    def mock_create_ad_response_model(self):
        """Mock create ad response model for testing."""
        mock = Mock(spec=CreateAdResponse)
        mock.success = True
        mock.ad_id = "test-ad-123"
        mock.message = "Ad created successfully"
        mock.ad_data = {
            "id": "test-ad-123",
            "title": "Test Advertisement",
            "description": "A test advertisement for testing purposes",
            "status": "draft"
        }
        
        # Mock methods
        mock.to_dict.return_value = {
            "success": mock.success,
            "ad_id": mock.ad_id,
            "message": mock.message,
            "ad_data": mock.ad_data
        }
        
        return mock

    @pytest.fixture
    def mock_optimization_request_model(self):
        """Mock optimization request model for testing."""
        mock = Mock(spec=OptimizationRequest)
        mock.ad_id = "test-ad-123"
        mock.optimization_type = "performance"
        mock.target_metrics = ["ctr", "conversion_rate"]
        mock.constraints = {
            "max_budget": 1500.0,
            "min_impressions": 500
        }
        mock.optimization_level = "aggressive"
        
        # Mock methods
        mock.validate.return_value = True
        mock.to_dict.return_value = {
            "ad_id": mock.ad_id,
            "optimization_type": mock.optimization_type,
            "target_metrics": mock.target_metrics
        }
        
        return mock

    @pytest.fixture
    def mock_optimization_response_model(self):
        """Mock optimization response model for testing."""
        mock = Mock(spec=OptimizationResponse)
        mock.success = True
        mock.optimization_id = "test-optimization-123"
        mock.message = "Optimization completed successfully"
        mock.improvements = {
            "ctr": 0.15,
            "conversion_rate": 0.12,
            "roas": 3.2
        }
        mock.recommendations = [
            "Increase bid by 15%",
            "Refine targeting criteria",
            "Update ad creative"
        ]
        
        # Mock methods
        mock.to_dict.return_value = {
            "success": mock.success,
            "optimization_id": mock.optimization_id,
            "message": mock.message,
            "improvements": mock.improvements
        }
        
        return mock

    @pytest.fixture
    def mock_database_session(self):
        """Mock database session for testing."""
        mock = AsyncMock()
        mock.execute.return_value = AsyncMock()
        mock.commit.return_value = None
        mock.rollback.return_value = None
        mock.close.return_value = None
        
        # Mock query results
        mock_result = AsyncMock()
        mock_result.scalars.return_value = [Mock(), Mock()]
        mock_result.first.return_value = Mock()
        mock.execute.return_value = mock_result
        
        return mock

    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing."""
        mock = AsyncMock()
        mock.create.return_value = Mock()
        mock.get_by_id.return_value = Mock()
        mock.update.return_value = Mock()
        mock.delete.return_value = True
        mock.list.return_value = [Mock(), Mock(), Mock()]
        mock.count.return_value = 3
        
        return mock

    @pytest.fixture
    def mock_service(self):
        """Mock service for testing."""
        mock = AsyncMock()
        mock.create_ad.return_value = Mock()
        mock.get_ad.return_value = Mock()
        mock.update_ad.return_value = Mock()
        mock.delete_ad.return_value = True
        mock.list_ads.return_value = [Mock(), Mock(), Mock()]
        
        return mock

    @pytest.fixture
    def mock_use_case(self):
        """Mock use case for testing."""
        mock = AsyncMock()
        mock.execute.return_value = Mock()
        
        return mock

    @pytest.fixture
    def mock_external_service(self):
        """Mock external service for testing."""
        mock = AsyncMock()
        mock.call_api.return_value = {"status": "success", "data": "test_data"}
        mock.validate_response.return_value = True
        mock.handle_error.return_value = {"error": "handled"}
        
        return mock

    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service for testing."""
        mock = AsyncMock()
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        mock.exists.return_value = False
        mock.expire.return_value = True
        
        return mock

    @pytest.fixture
    def mock_storage_service(self):
        """Mock storage service for testing."""
        mock = AsyncMock()
        mock.save_file.return_value = "https://example.com/file.jpg"
        mock.get_file.return_value = b"file_content"
        mock.delete_file.return_value = True
        mock.file_exists.return_value = True
        mock.get_file_url.return_value = "https://example.com/file.jpg"
        
        return mock

    @pytest.fixture
    def mock_optimization_service(self):
        """Mock optimization service for testing."""
        mock = AsyncMock()
        mock.optimize_ad.return_value = {"improvements": {"ctr": 0.15}}
        mock.get_optimization_history.return_value = [Mock(), Mock()]
        mock.revert_optimization.return_value = True
        
        return mock

    @pytest.fixture
    def mock_training_service(self):
        """Mock training service for testing."""
        mock = AsyncMock()
        mock.train_model.return_value = {"model_id": "test-model-123"}
        mock.get_training_status.return_value = "completed"
        mock.evaluate_model.return_value = {"accuracy": 0.85}
        
        return mock

    @pytest.fixture
    def mock_analytics_service(self):
        """Mock analytics service for testing."""
        mock = AsyncMock()
        mock.get_ad_performance.return_value = {"ctr": 0.05, "conversions": 10}
        mock.get_campaign_analytics.return_value = {"total_spend": 5000.0, "roas": 2.5}
        mock.generate_report.return_value = {"report_id": "test-report-123"}
        
        return mock

    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service for testing."""
        mock = AsyncMock()
        mock.send_notification.return_value = {"notification_id": "test-notification-123"}
        mock.schedule_notification.return_value = True
        mock.cancel_notification.return_value = True
        
        return mock

    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service for testing."""
        mock = AsyncMock()
        mock.validate_ad_data.return_value = {"is_valid": True, "errors": []}
        mock.validate_campaign_data.return_value = {"is_valid": True, "errors": []}
        mock.validate_targeting_criteria.return_value = {"is_valid": True, "conflicts": []}
        
        return mock

    @pytest.fixture
    def mock_authorization_service(self):
        """Mock authorization service for testing."""
        mock = AsyncMock()
        mock.check_permission.return_value = True
        mock.get_user_roles.return_value = ["admin", "ad_manager"]
        mock.validate_token.return_value = {"user_id": "test-user-123", "valid": True}
        
        return mock

    @pytest.fixture
    def mock_rate_limiter(self):
        """Mock rate limiter for testing."""
        mock = AsyncMock()
        mock.check_rate_limit.return_value = True
        mock.increment_request_count.return_value = 1
        mock.get_remaining_requests.return_value = 99
        
        return mock

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        mock = Mock()
        mock.info.return_value = None
        mock.warning.return_value = None
        mock.error.return_value = None
        mock.debug.return_value = None
        mock.critical.return_value = None
        
        return mock

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector for testing."""
        mock = AsyncMock()
        mock.record_metric.return_value = None
        mock.get_metrics.return_value = {"requests": 100, "errors": 2}
        mock.reset_metrics.return_value = None
        
        return mock

    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for testing."""
        mock = AsyncMock()
        mock.get_config.return_value = {"database_url": "sqlite:///test.db"}
        mock.update_config.return_value = True
        mock.reload_config.return_value = None
        
        return mock

    @pytest.fixture
    def mock_error_handler(self):
        """Mock error handler for testing."""
        mock = AsyncMock()
        mock.handle_error.return_value = {"error_handled": True, "recovery_action": "retry"}
        mock.log_error.return_value = None
        mock.notify_admin.return_value = True
        
        return mock

    @pytest.fixture
    def mock_background_task(self):
        """Mock background task for testing."""
        mock = AsyncMock()
        mock.schedule_task.return_value = {"task_id": "test-task-123"}
        mock.get_task_status.return_value = "running"
        mock.cancel_task.return_value = True
        
        return mock

    @pytest.fixture
    def mock_file_processor(self):
        """Mock file processor for testing."""
        mock = AsyncMock()
        mock.process_image.return_value = {"processed_url": "https://example.com/processed.jpg"}
        mock.process_video.return_value = {"processed_url": "https://example.com/processed.mp4"}
        mock.validate_file.return_value = {"is_valid": True, "file_type": "image/jpeg"}
        
        return mock

    @pytest.fixture
    def mock_api_client(self):
        """Mock API client for testing."""
        mock = AsyncMock()
        mock.get.return_value = {"status": 200, "data": "test_data"}
        mock.post.return_value = {"status": 201, "data": "created_data"}
        mock.put.return_value = {"status": 200, "data": "updated_data"}
        mock.delete.return_value = {"status": 204}
        
        return mock

    @pytest.fixture
    def mock_database_connection(self):
        """Mock database connection for testing."""
        mock = AsyncMock()
        mock.execute.return_value = AsyncMock()
        mock.fetchone.return_value = {"id": 1, "name": "test"}
        mock.fetchall.return_value = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        mock.close.return_value = None
        
        return mock

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        mock = AsyncMock()
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.exists.return_value = False
        mock.expire.return_value = True
        
        return mock

    @pytest.fixture
    def mock_elasticsearch_client(self):
        """Mock Elasticsearch client for testing."""
        mock = AsyncMock()
        mock.index.return_value = {"_id": "test-id-123", "result": "created"}
        mock.search.return_value = {"hits": {"total": {"value": 1}, "hits": [{"_source": {"title": "test"}}]}}
        mock.update.return_value = {"_id": "test-id-123", "result": "updated"}
        mock.delete.return_value = {"_id": "test-id-123", "result": "deleted"}
        
        return mock

    @pytest.fixture
    def mock_queue_manager(self):
        """Mock queue manager for testing."""
        mock = AsyncMock()
        mock.enqueue_task.return_value = {"task_id": "test-task-123"}
        mock.dequeue_task.return_value = {"task_data": "test_data"}
        mock.get_queue_status.return_value = {"pending": 5, "processing": 2}
        
        return mock

    @pytest.fixture
    def mock_event_bus(self):
        """Mock event bus for testing."""
        mock = AsyncMock()
        mock.publish_event.return_value = True
        mock.subscribe_to_event.return_value = {"subscription_id": "test-sub-123"}
        mock.unsubscribe_from_event.return_value = True
        
        return mock

    @pytest.fixture
    def mock_health_checker(self):
        """Mock health checker for testing."""
        mock = AsyncMock()
        mock.check_health.return_value = {"status": "healthy", "details": {"database": "ok", "cache": "ok"}}
        mock.get_health_status.return_value = "healthy"
        mock.run_health_check.return_value = {"overall_status": "healthy"}
        
        return mock

    @pytest.fixture
    def mock_security_manager(self):
        """Mock security manager for testing."""
        mock = AsyncMock()
        mock.encrypt_data.return_value = "encrypted_data"
        mock.decrypt_data.return_value = "decrypted_data"
        mock.hash_password.return_value = "hashed_password"
        mock.verify_password.return_value = True
        
        return mock

    @pytest.fixture
    def mock_audit_logger(self):
        """Mock audit logger for testing."""
        mock = AsyncMock()
        mock.log_action.return_value = {"log_id": "test-log-123"}
        mock.get_audit_trail.return_value = [{"action": "create", "user": "test-user", "timestamp": datetime.now()}]
        mock.export_audit_log.return_value = {"export_id": "test-export-123"}
        
        return mock


# Utility functions for creating mock models
@pytest.fixture
def mock_model_factory():
    """Factory for creating mock models."""
    
    def create_mock_ad(**kwargs):
        """Create a mock ad model with custom attributes."""
        mock = Mock(spec=Ad)
        mock.id = kwargs.get("id", f"ad-{uuid.uuid4().hex[:8]}")
        mock.title = kwargs.get("title", "Test Ad")
        mock.description = kwargs.get("description", "Test Description")
        mock.status = kwargs.get("status", "draft")
        mock.platform = kwargs.get("platform", "facebook")
        mock.budget = kwargs.get("budget", 1000.0)
        
        # Mock methods
        mock.approve.return_value = mock
        mock.activate.return_value = mock
        mock.pause.return_value = mock
        mock.archive.return_value = mock
        
        return mock
    
    def create_mock_campaign(**kwargs):
        """Create a mock campaign model with custom attributes."""
        mock = Mock(spec=AdCampaign)
        mock.id = kwargs.get("id", f"campaign-{uuid.uuid4().hex[:8]}")
        mock.name = kwargs.get("name", "Test Campaign")
        mock.description = kwargs.get("description", "Test Campaign Description")
        mock.status = kwargs.get("status", "active")
        mock.budget = kwargs.get("budget", 5000.0)
        
        # Mock methods
        mock.activate.return_value = mock
        mock.pause.return_value = mock
        mock.archive.return_value = mock
        
        return mock
    
    def create_mock_request(**kwargs):
        """Create a mock request model with custom attributes."""
        mock = Mock()
        mock.title = kwargs.get("title", "Test Request")
        mock.description = kwargs.get("description", "Test Description")
        mock.platform = kwargs.get("platform", "facebook")
        mock.budget = kwargs.get("budget", 1000.0)
        
        # Mock methods
        mock.validate.return_value = kwargs.get("is_valid", True)
        mock.to_dict.return_value = kwargs
        
        return mock
    
    def create_mock_response(**kwargs):
        """Create a mock response model with custom attributes."""
        mock = Mock()
        mock.success = kwargs.get("success", True)
        mock.message = kwargs.get("message", "Success")
        mock.data = kwargs.get("data", {})
        
        # Mock methods
        mock.to_dict.return_value = kwargs
        
        return mock
    
    return {
        "create_mock_ad": create_mock_ad,
        "create_mock_campaign": create_mock_campaign,
        "create_mock_request": create_mock_request,
        "create_mock_response": create_mock_response
    }


# Test scenario models
@pytest.fixture
def test_scenario_models():
    """Models for different test scenarios."""
    
    def get_happy_path_models():
        """Get models for happy path testing."""
        return {
            "ad": Mock(id="happy-ad-123", title="Happy Path Ad", status="draft"),
            "campaign": Mock(id="happy-campaign-123", name="Happy Campaign", status="active"),
            "request": Mock(title="Happy Request", platform="facebook", budget=1000.0),
            "response": Mock(success=True, message="Success")
        }
    
    def get_error_scenario_models():
        """Get models for error scenario testing."""
        return {
            "ad": Mock(id="error-ad-123", title="Error Ad", status="error"),
            "campaign": Mock(id="error-campaign-123", name="Error Campaign", status="paused"),
            "request": Mock(title="", platform="invalid", budget=-100),
            "response": Mock(success=False, message="Validation failed")
        }
    
    def get_edge_case_models():
        """Get models for edge case testing."""
        return {
            "ad": Mock(id="edge-ad-123", title="A" * 1000, status="draft"),
            "campaign": Mock(id="edge-campaign-123", name="Edge Campaign", budget=0.01),
            "request": Mock(title="Edge", platform="facebook", budget=999999.99),
            "response": Mock(success=True, message="Edge case handled")
        }
    
    return {
        "get_happy_path_models": get_happy_path_models,
        "get_error_scenario_models": get_error_scenario_models,
        "get_edge_case_models": get_edge_case_models
    }

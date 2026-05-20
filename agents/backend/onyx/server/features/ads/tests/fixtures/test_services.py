"""
Test service fixtures for the ads feature.

This module provides comprehensive test service fixtures for:
- Domain services (AdService, CampaignService, OptimizationService)
- Application services (Use Cases)
- Infrastructure services (Database, Storage, Cache)
- External services (AI providers, analytics, notifications)
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

# Import domain services
from agents.backend.onyx.server.features.ads.domain.services import AdService, CampaignService, OptimizationService
from agents.backend.onyx.server.features.ads.application.use_cases import (
    CreateAdUseCase, ApproveAdUseCase, ActivateAdUseCase, PauseAdUseCase, ArchiveAdUseCase,
    CreateCampaignUseCase, ActivateCampaignUseCase, PauseCampaignUseCase, OptimizeAdUseCase
)


class TestServiceFixtures:
    """Test service fixtures for ads feature testing."""

    @pytest.fixture
    def mock_ad_service(self):
        """Mock ad service for testing."""
        mock = AsyncMock(spec=AdService)
        
        # Mock ad creation
        mock_ad = Mock()
        mock_ad.id = "test-ad-123"
        mock_ad.title = "Test Advertisement"
        mock_ad.status = "draft"
        mock_ad.platform = "facebook"
        mock_ad.budget = 1000.0
        
        mock.create_ad.return_value = mock_ad
        mock.get_ad.return_value = mock_ad
        mock.update_ad.return_value = mock_ad
        mock.delete_ad.return_value = True
        mock.list_ads.return_value = [mock_ad, Mock(), Mock()]
        mock.count_ads.return_value = 3
        
        # Mock status change methods
        mock.approve_ad.return_value = mock_ad
        mock.activate_ad.return_value = mock_ad
        mock.pause_ad.return_value = mock_ad
        mock.archive_ad.return_value = mock_ad
        
        # Mock validation methods
        mock.validate_ad_data.return_value = {"is_valid": True, "errors": []}
        mock.validate_targeting_criteria.return_value = {"is_valid": True, "conflicts": []}
        
        return mock

    @pytest.fixture
    def mock_campaign_service(self):
        """Mock campaign service for testing."""
        mock = AsyncMock(spec=CampaignService)
        
        # Mock campaign creation
        mock_campaign = Mock()
        mock_campaign.id = "test-campaign-123"
        mock_campaign.name = "Test Campaign"
        mock_campaign.status = "active"
        mock_campaign.budget = 5000.0
        
        mock.create_campaign.return_value = mock_campaign
        mock.get_campaign.return_value = mock_campaign
        mock.update_campaign.return_value = mock_campaign
        mock.delete_campaign.return_value = True
        mock.list_campaigns.return_value = [mock_campaign, Mock(), Mock()]
        mock.count_campaigns.return_value = 3
        
        # Mock status change methods
        mock.activate_campaign.return_value = mock_campaign
        mock.pause_campaign.return_value = mock_campaign
        mock.archive_campaign.return_value = mock_campaign
        
        # Mock budget methods
        mock.update_budget.return_value = mock_campaign
        mock.check_budget_availability.return_value = True
        mock.get_budget_utilization.return_value = 0.6
        
        return mock

    @pytest.fixture
    def mock_optimization_service(self):
        """Mock optimization service for testing."""
        mock = AsyncMock(spec=OptimizationService)
        
        # Mock optimization methods
        mock.optimize_ad.return_value = {
            "optimization_id": "test-optimization-123",
            "improvements": {"ctr": 0.15, "conversion_rate": 0.12},
            "recommendations": ["Increase bid by 15%", "Refine targeting"]
        }
        mock.get_optimization_history.return_value = [Mock(), Mock()]
        mock.revert_optimization.return_value = True
        mock.get_optimization_status.return_value = "completed"
        
        # Mock performance analysis
        mock.analyze_performance.return_value = {
            "current_metrics": {"ctr": 0.05, "conversion_rate": 0.08},
            "benchmark_metrics": {"ctr": 0.12, "conversion_rate": 0.15},
            "improvement_potential": {"ctr": 0.07, "conversion_rate": 0.07}
        }
        
        # Mock A/B testing
        mock.create_ab_test.return_value = {"test_id": "test-ab-123"}
        mock.get_ab_test_results.return_value = {"variant_a": 0.06, "variant_b": 0.08}
        
        return mock

    @pytest.fixture
    def mock_create_ad_use_case(self):
        """Mock create ad use case for testing."""
        mock = AsyncMock(spec=CreateAdUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.ad_id = "test-ad-123"
        mock_response.message = "Ad created successfully"
        mock_response.ad_data = {"id": "test-ad-123", "title": "Test Ad"}
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_approve_ad_use_case(self):
        """Mock approve ad use case for testing."""
        mock = AsyncMock(spec=ApproveAdUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.ad_id = "test-ad-123"
        mock_response.message = "Ad approved successfully"
        mock_response.status = "approved"
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_activate_ad_use_case(self):
        """Mock activate ad use case for testing."""
        mock = AsyncMock(spec=ActivateAdUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.ad_id = "test-ad-123"
        mock_response.message = "Ad activated successfully"
        mock_response.status = "active"
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_pause_ad_use_case(self):
        """Mock pause ad use case for testing."""
        mock = AsyncMock(spec=PauseAdUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.ad_id = "test-ad-123"
        mock_response.message = "Ad paused successfully"
        mock_response.status = "paused"
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_archive_ad_use_case(self):
        """Mock archive ad use case for testing."""
        mock = AsyncMock(spec=ArchiveAdUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.ad_id = "test-ad-123"
        mock_response.message = "Ad archived successfully"
        mock_response.status = "archived"
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_create_campaign_use_case(self):
        """Mock create campaign use case for testing."""
        mock = AsyncMock(spec=CreateCampaignUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.campaign_id = "test-campaign-123"
        mock_response.message = "Campaign created successfully"
        mock_response.campaign_data = {"id": "test-campaign-123", "name": "Test Campaign"}
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_activate_campaign_use_case(self):
        """Mock activate campaign use case for testing."""
        mock = AsyncMock(spec=ActivateCampaignUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.campaign_id = "test-campaign-123"
        mock_response.message = "Campaign activated successfully"
        mock_response.status = "active"
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_pause_campaign_use_case(self):
        """Mock pause campaign use case for testing."""
        mock = AsyncMock(spec=PauseCampaignUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.campaign_id = "test-campaign-123"
        mock_response.message = "Campaign paused successfully"
        mock_response.status = "paused"
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_optimize_ad_use_case(self):
        """Mock optimize ad use case for testing."""
        mock = AsyncMock(spec=OptimizeAdUseCase)
        
        # Mock successful execution
        mock_response = Mock()
        mock_response.success = True
        mock_response.optimization_id = "test-optimization-123"
        mock_response.message = "Ad optimization completed successfully"
        mock_response.improvements = {"ctr": 0.15, "conversion_rate": 0.12}
        mock_response.recommendations = ["Increase bid by 15%", "Refine targeting"]
        
        mock.execute.return_value = mock_response
        
        return mock

    @pytest.fixture
    def mock_database_service(self):
        """Mock database service for testing."""
        mock = AsyncMock()
        
        # Mock session management
        mock_session = AsyncMock()
        mock_session.execute.return_value = AsyncMock()
        mock_session.commit.return_value = None
        mock_session.rollback.return_value = None
        mock_session.close.return_value = None
        
        mock.get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock connection management
        mock.get_connection.return_value = AsyncMock()
        mock.return_connection.return_value = None
        mock.get_connection_stats.return_value = {
            "pool_size": 5,
            "checked_in": 3,
            "checked_out": 2
        }
        
        # Mock health checks
        mock.check_health.return_value = {"status": "healthy", "details": {"database": "ok"}}
        mock.get_health_status.return_value = "healthy"
        
        return mock

    @pytest.fixture
    def mock_storage_service(self):
        """Mock storage service for testing."""
        mock = AsyncMock()
        
        # Mock file operations
        mock.save_file.return_value = "https://example.com/stored_file.jpg"
        mock.get_file.return_value = b"file_content_bytes"
        mock.delete_file.return_value = True
        mock.file_exists.return_value = True
        mock.get_file_url.return_value = "https://example.com/file.jpg"
        
        # Mock directory operations
        mock.create_directory.return_value = True
        mock.list_files.return_value = ["file1.jpg", "file2.jpg"]
        mock.get_directory_size.return_value = 1024 * 1024  # 1MB
        
        # Mock validation
        mock.validate_file.return_value = {"is_valid": True, "file_type": "image/jpeg"}
        mock.get_file_info.return_value = {
            "size": 1024,
            "type": "image/jpeg",
            "dimensions": {"width": 800, "height": 600}
        }
        
        return mock

    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service for testing."""
        mock = AsyncMock()
        
        # Mock basic operations
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        mock.exists.return_value = False
        mock.expire.return_value = True
        
        # Mock advanced operations
        mock.get_many.return_value = {"key1": "value1", "key2": "value2"}
        mock.set_many.return_value = True
        mock.delete_many.return_value = 2
        mock.clear.return_value = True
        
        # Mock statistics
        mock.get_stats.return_value = {
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.83,
            "memory_usage": 1024 * 1024
        }
        
        # Mock pattern operations
        mock.get_by_pattern.return_value = ["key1", "key2"]
        mock.delete_by_pattern.return_value = 2
        
        return mock

    @pytest.fixture
    def mock_external_service(self):
        """Mock external service for testing."""
        mock = AsyncMock()
        
        # Mock API calls
        mock.call_api.return_value = {"status": "success", "data": "test_data"}
        mock.get.return_value = {"status": 200, "data": "test_data"}
        mock.post.return_value = {"status": 201, "data": "created_data"}
        mock.put.return_value = {"status": 200, "data": "updated_data"}
        mock.delete.return_value = {"status": 204}
        
        # Mock response handling
        mock.validate_response.return_value = True
        mock.handle_error.return_value = {"error": "handled", "recovery_action": "retry"}
        mock.parse_response.return_value = {"parsed": "data"}
        
        # Mock rate limiting
        mock.check_rate_limit.return_value = True
        mock.get_rate_limit_info.return_value = {"remaining": 99, "reset_time": datetime.now() + timedelta(hours=1)}
        
        return mock

    @pytest.fixture
    def mock_ai_provider_service(self):
        """Mock AI provider service for testing."""
        mock = AsyncMock()
        
        # Mock text generation
        mock.generate_text.return_value = {
            "generated_text": "This is AI-generated ad copy",
            "confidence": 0.85,
            "tokens_used": 25
        }
        
        # Mock content optimization
        mock.optimize_content.return_value = {
            "optimized_content": "Optimized ad copy",
            "improvements": ["Better CTA", "Clearer messaging"],
            "score": 0.92
        }
        
        # Mock audience analysis
        mock.analyze_audience.return_value = {
            "audience_insights": ["Tech-savvy", "Budget-conscious"],
            "recommendations": ["Use technical terms", "Emphasize value"],
            "confidence": 0.78
        }
        
        # Mock performance prediction
        mock.predict_performance.return_value = {
            "predicted_ctr": 0.08,
            "predicted_conversion_rate": 0.12,
            "confidence_interval": [0.06, 0.10]
        }
        
        return mock

    @pytest.fixture
    def mock_analytics_service(self):
        """Mock analytics service for testing."""
        mock = AsyncMock()
        
        # Mock performance metrics
        mock.get_ad_performance.return_value = {
            "ctr": 0.05,
            "conversions": 10,
            "spend": 150.0,
            "roas": 2.5
        }
        
        mock.get_campaign_analytics.return_value = {
            "total_spend": 5000.0,
            "total_conversions": 50,
            "overall_roas": 2.8,
            "performance_trend": "improving"
        }
        
        # Mock reporting
        mock.generate_report.return_value = {
            "report_id": "test-report-123",
            "report_url": "https://example.com/report.pdf",
            "generated_at": datetime.now()
        }
        
        mock.get_report_status.return_value = "completed"
        mock.export_report.return_value = {"export_id": "test-export-123"}
        
        # Mock insights
        mock.get_insights.return_value = [
            {"type": "performance", "message": "CTR improved by 15%", "confidence": 0.85},
            {"type": "audience", "message": "Mobile users convert better", "confidence": 0.78}
        ]
        
        return mock

    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service for testing."""
        mock = AsyncMock()
        
        # Mock notification sending
        mock.send_notification.return_value = {
            "notification_id": "test-notification-123",
            "status": "sent",
            "sent_at": datetime.now()
        }
        
        # Mock scheduling
        mock.schedule_notification.return_value = {
            "scheduled_id": "test-scheduled-123",
            "scheduled_for": datetime.now() + timedelta(hours=1)
        }
        
        mock.cancel_notification.return_value = True
        
        # Mock notification types
        mock.send_email.return_value = {"email_id": "test-email-123"}
        mock.send_sms.return_value = {"sms_id": "test-sms-123"}
        mock.send_push.return_value = {"push_id": "test-push-123"}
        
        # Mock notification history
        mock.get_notification_history.return_value = [
            {"id": "notif-1", "type": "email", "status": "sent"},
            {"id": "notif-2", "type": "sms", "status": "delivered"}
        ]
        
        return mock

    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service for testing."""
        mock = AsyncMock()
        
        # Mock data validation
        mock.validate_ad_data.return_value = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        mock.validate_campaign_data.return_value = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        mock.validate_targeting_criteria.return_value = {
            "is_valid": True,
            "conflicts": [],
            "recommendations": []
        }
        
        # Mock business rule validation
        mock.validate_business_rules.return_value = {
            "rules_passed": 5,
            "rules_failed": 0,
            "violations": []
        }
        
        # Mock compliance validation
        mock.validate_compliance.return_value = {
            "is_compliant": True,
            "compliance_score": 0.95,
            "issues": []
        }
        
        return mock

    @pytest.fixture
    def mock_authorization_service(self):
        """Mock authorization service for testing."""
        mock = AsyncMock()
        
        # Mock permission checking
        mock.check_permission.return_value = True
        mock.check_permissions.return_value = {"create": True, "read": True, "update": False, "delete": False}
        
        # Mock role management
        mock.get_user_roles.return_value = ["admin", "ad_manager"]
        mock.has_role.return_value = True
        mock.has_any_role.return_value = True
        
        # Mock token validation
        mock.validate_token.return_value = {
            "user_id": "test-user-123",
            "valid": True,
            "expires_at": datetime.now() + timedelta(hours=1)
        }
        
        # Mock access control
        mock.can_access_resource.return_value = True
        mock.get_resource_permissions.return_value = {
            "resource_id": "test-resource-123",
            "permissions": ["read", "write"]
        }
        
        return mock

    @pytest.fixture
    def mock_rate_limiter_service(self):
        """Mock rate limiter service for testing."""
        mock = AsyncMock()
        
        # Mock rate limit checking
        mock.check_rate_limit.return_value = True
        mock.is_rate_limited.return_value = False
        
        # Mock request counting
        mock.increment_request_count.return_value = 1
        mock.get_request_count.return_value = 5
        mock.get_remaining_requests.return_value = 95
        
        # Mock rate limit info
        mock.get_rate_limit_info.return_value = {
            "limit": 100,
            "remaining": 95,
            "reset_time": datetime.now() + timedelta(hours=1),
            "window_size": 3600
        }
        
        # Mock rate limit management
        mock.set_rate_limit.return_value = True
        mock.reset_rate_limit.return_value = True
        mock.get_rate_limit_stats.return_value = {
            "total_requests": 1000,
            "rate_limited_requests": 50,
            "average_requests_per_minute": 10
        }
        
        return mock

    @pytest.fixture
    def mock_logging_service(self):
        """Mock logging service for testing."""
        mock = AsyncMock()
        
        # Mock log levels
        mock.info.return_value = None
        mock.warning.return_value = None
        mock.error.return_value = None
        mock.debug.return_value = None
        mock.critical.return_value = None
        
        # Mock structured logging
        mock.log_with_context.return_value = None
        mock.log_with_metadata.return_value = None
        
        # Mock log retrieval
        mock.get_logs.return_value = [
            {"level": "INFO", "message": "Test log message", "timestamp": datetime.now()}
        ]
        
        mock.search_logs.return_value = [
            {"level": "ERROR", "message": "Error occurred", "timestamp": datetime.now()}
        ]
        
        # Mock log export
        mock.export_logs.return_value = {"export_id": "test-log-export-123"}
        
        return mock

    @pytest.fixture
    def mock_metrics_service(self):
        """Mock metrics service for testing."""
        mock = AsyncMock()
        
        # Mock metric recording
        mock.record_metric.return_value = None
        mock.record_metrics.return_value = None
        mock.increment_counter.return_value = None
        mock.set_gauge.return_value = None
        
        # Mock metric retrieval
        mock.get_metric.return_value = 100
        mock.get_metrics.return_value = {"requests": 100, "errors": 2, "latency": 150}
        
        # Mock metric aggregation
        mock.get_metric_summary.return_value = {
            "total_requests": 1000,
            "average_latency": 150.5,
            "error_rate": 0.02
        }
        
        # Mock metric export
        mock.export_metrics.return_value = {"export_id": "test-metrics-export-123"}
        
        return mock

    @pytest.fixture
    def mock_configuration_service(self):
        """Mock configuration service for testing."""
        mock = AsyncMock()
        
        # Mock config retrieval
        mock.get_config.return_value = {
            "database_url": "sqlite:///test.db",
            "redis_url": "redis://localhost:6379",
            "api_key": "test-api-key"
        }
        
        mock.get_config_value.return_value = "test-value"
        mock.has_config.return_value = True
        
        # Mock config management
        mock.update_config.return_value = True
        mock.reload_config.return_value = None
        mock.reset_config.return_value = None
        
        # Mock config validation
        mock.validate_config.return_value = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Mock config export
        mock.export_config.return_value = {"export_id": "test-config-export-123"}
        
        return mock

    @pytest.fixture
    def mock_error_handling_service(self):
        """Mock error handling service for testing."""
        mock = AsyncMock()
        
        # Mock error handling
        mock.handle_error.return_value = {
            "error_handled": True,
            "recovery_action": "retry",
            "error_id": "test-error-123"
        }
        
        # Mock error logging
        mock.log_error.return_value = None
        mock.log_error_with_context.return_value = None
        
        # Mock error reporting
        mock.report_error.return_value = {"report_id": "test-error-report-123"}
        mock.notify_admin.return_value = True
        
        # Mock error recovery
        mock.attempt_recovery.return_value = {
            "recovery_attempted": True,
            "success": True,
            "actions_taken": ["retry", "fallback"]
        }
        
        # Mock error statistics
        mock.get_error_stats.return_value = {
            "total_errors": 50,
            "resolved_errors": 45,
            "error_rate": 0.05
        }
        
        return mock

    @pytest.fixture
    def mock_background_task_service(self):
        """Mock background task service for testing."""
        mock = AsyncMock()
        
        # Mock task scheduling
        mock.schedule_task.return_value = {
            "task_id": "test-task-123",
            "scheduled_at": datetime.now(),
            "estimated_duration": 300
        }
        
        # Mock task management
        mock.get_task_status.return_value = "running"
        mock.cancel_task.return_value = True
        mock.pause_task.return_value = True
        mock.resume_task.return_value = True
        
        # Mock task monitoring
        mock.get_task_progress.return_value = {
            "progress": 0.75,
            "current_step": "Processing data",
            "estimated_completion": datetime.now() + timedelta(minutes=5)
        }
        
        # Mock task history
        mock.get_task_history.return_value = [
            {"task_id": "task-1", "status": "completed", "duration": 120},
            {"task_id": "task-2", "status": "failed", "duration": 60}
        ]
        
        return mock

    @pytest.fixture
    def mock_file_processing_service(self):
        """Mock file processing service for testing."""
        mock = AsyncMock()
        
        # Mock image processing
        mock.process_image.return_value = {
            "processed_url": "https://example.com/processed.jpg",
            "format": "jpeg",
            "dimensions": {"width": 800, "height": 600},
            "file_size": 102400
        }
        
        # Mock video processing
        mock.process_video.return_value = {
            "processed_url": "https://example.com/processed.mp4",
            "format": "mp4",
            "duration": 30,
            "file_size": 2048000
        }
        
        # Mock file validation
        mock.validate_file.return_value = {
            "is_valid": True,
            "file_type": "image/jpeg",
            "file_size": 102400,
            "dimensions": {"width": 800, "height": 600}
        }
        
        # Mock file optimization
        mock.optimize_file.return_value = {
            "optimized_url": "https://example.com/optimized.jpg",
            "compression_ratio": 0.7,
            "quality_score": 0.9
        }
        
        return mock

    @pytest.fixture
    def mock_api_client_service(self):
        """Mock API client service for testing."""
        mock = AsyncMock()
        
        # Mock HTTP methods
        mock.get.return_value = {"status": 200, "data": "test_data"}
        mock.post.return_value = {"status": 201, "data": "created_data"}
        mock.put.return_value = {"status": 200, "data": "updated_data"}
        mock.delete.return_value = {"status": 204}
        mock.patch.return_value = {"status": 200, "data": "patched_data"}
        
        # Mock request/response handling
        mock.build_request.return_value = {"url": "https://api.example.com", "headers": {}}
        mock.parse_response.return_value = {"parsed": "data"}
        mock.handle_error_response.return_value = {"error": "handled"}
        
        # Mock authentication
        mock.authenticate.return_value = {"token": "test-token", "expires_at": datetime.now() + timedelta(hours=1)}
        mock.refresh_token.return_value = {"token": "new-token", "expires_at": datetime.now() + timedelta(hours=1)}
        
        # Mock rate limiting
        mock.check_rate_limit.return_value = True
        mock.get_rate_limit_info.return_value = {"remaining": 99, "reset_time": datetime.now() + timedelta(hours=1)}
        
        return mock

    @pytest.fixture
    def mock_database_connection_service(self):
        """Mock database connection service for testing."""
        mock = AsyncMock()
        
        # Mock connection management
        mock.get_connection.return_value = AsyncMock()
        mock.return_connection.return_value = None
        mock.close_connection.return_value = None
        
        # Mock connection pooling
        mock.get_pool_stats.return_value = {
            "pool_size": 10,
            "active_connections": 3,
            "idle_connections": 7,
            "waiting_requests": 0
        }
        
        # Mock connection health
        mock.check_connection_health.return_value = {
            "status": "healthy",
            "response_time": 5,
            "last_check": datetime.now()
        }
        
        # Mock connection monitoring
        mock.monitor_connections.return_value = {
            "total_connections": 10,
            "failed_connections": 0,
            "connection_errors": []
        }
        
        return mock

    @pytest.fixture
    def mock_redis_client_service(self):
        """Mock Redis client service for testing."""
        mock = AsyncMock()
        
        # Mock basic operations
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.exists.return_value = False
        mock.expire.return_value = True
        
        # Mock advanced operations
        mock.get_many.return_value = {"key1": "value1", "key2": "value2"}
        mock.set_many.return_value = True
        mock.delete_many.return_value = 2
        mock.clear.return_value = True
        
        # Mock data structures
        mock.hget.return_value = "field_value"
        mock.hset.return_value = 1
        mock.hgetall.return_value = {"field1": "value1", "field2": "value2"}
        
        mock.lpush.return_value = 1
        mock.rpop.return_value = "popped_value"
        mock.lrange.return_value = ["value1", "value2", "value3"]
        
        # Mock pub/sub
        mock.publish.return_value = 1
        mock.subscribe.return_value = AsyncMock()
        
        return mock

    @pytest.fixture
    def mock_elasticsearch_client_service(self):
        """Mock Elasticsearch client service for testing."""
        mock = AsyncMock()
        
        # Mock document operations
        mock.index.return_value = {"_id": "test-id-123", "result": "created"}
        mock.get.return_value = {"_id": "test-id-123", "_source": {"title": "test"}}
        mock.update.return_value = {"_id": "test-id-123", "result": "updated"}
        mock.delete.return_value = {"_id": "test-id-123", "result": "deleted"}
        
        # Mock search operations
        mock.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{"_source": {"title": "test"}}]
            }
        }
        
        mock.count.return_value = {"count": 1}
        mock.exists.return_value = True
        
        # Mock index management
        mock.create_index.return_value = {"acknowledged": True}
        mock.delete_index.return_value = {"acknowledged": True}
        mock.index_exists.return_value = True
        
        # Mock bulk operations
        mock.bulk.return_value = {"items": [{"index": {"_id": "test-id-123"}}]}
        
        return mock

    @pytest.fixture
    def mock_queue_manager_service(self):
        """Mock queue manager service for testing."""
        mock = AsyncMock()
        
        # Mock task management
        mock.enqueue_task.return_value = {"task_id": "test-task-123"}
        mock.dequeue_task.return_value = {"task_data": "test_data"}
        mock.get_task_status.return_value = "pending"
        
        # Mock queue management
        mock.create_queue.return_value = {"queue_id": "test-queue-123"}
        mock.delete_queue.return_value = True
        mock.purge_queue.return_value = True
        
        # Mock queue monitoring
        mock.get_queue_status.return_value = {
            "pending": 5,
            "processing": 2,
            "completed": 10,
            "failed": 1
        }
        
        mock.get_queue_stats.return_value = {
            "total_tasks": 18,
            "average_processing_time": 30,
            "success_rate": 0.95
        }
        
        return mock

    @pytest.fixture
    def mock_event_bus_service(self):
        """Mock event bus service for testing."""
        mock = AsyncMock()
        
        # Mock event publishing
        mock.publish_event.return_value = True
        mock.publish_events.return_value = [True, True, True]
        
        # Mock event subscription
        mock.subscribe_to_event.return_value = {"subscription_id": "test-sub-123"}
        mock.unsubscribe_from_event.return_value = True
        
        # Mock event handling
        mock.handle_event.return_value = {"handled": True, "result": "success"}
        mock.process_event.return_value = {"processed": True, "output": "event_output"}
        
        # Mock event history
        mock.get_event_history.return_value = [
            {"event_id": "event-1", "type": "ad_created", "timestamp": datetime.now()},
            {"event_id": "event-2", "type": "campaign_updated", "timestamp": datetime.now()}
        ]
        
        return mock

    @pytest.fixture
    def mock_health_checker_service(self):
        """Mock health checker service for testing."""
        mock = AsyncMock()
        
        # Mock health checks
        mock.check_health.return_value = {
            "status": "healthy",
            "details": {
                "database": "ok",
                "cache": "ok",
                "storage": "ok",
                "external_services": "ok"
            }
        }
        
        mock.get_health_status.return_value = "healthy"
        mock.run_health_check.return_value = {"overall_status": "healthy"}
        
        # Mock component health
        mock.check_database_health.return_value = {"status": "ok", "response_time": 5}
        mock.check_cache_health.return_value = {"status": "ok", "hit_rate": 0.85}
        mock.check_storage_health.return_value = {"status": "ok", "available_space": "10GB"}
        
        # Mock health monitoring
        mock.start_health_monitoring.return_value = True
        mock.stop_health_monitoring.return_value = True
        mock.get_health_metrics.return_value = {
            "uptime": 86400,
            "last_check": datetime.now(),
            "check_count": 100
        }
        
        return mock

    @pytest.fixture
    def mock_security_manager_service(self):
        """Mock security manager service for testing."""
        mock = AsyncMock()
        
        # Mock encryption/decryption
        mock.encrypt_data.return_value = "encrypted_data"
        mock.decrypt_data.return_value = "decrypted_data"
        mock.encrypt_file.return_value = {"encrypted_file": "encrypted.bin", "key_id": "key-123"}
        
        # Mock password management
        mock.hash_password.return_value = "hashed_password"
        mock.verify_password.return_value = True
        mock.generate_password.return_value = "generated_password"
        
        # Mock token management
        mock.generate_token.return_value = "generated_token"
        mock.validate_token.return_value = {"valid": True, "user_id": "test-user-123"}
        mock.refresh_token.return_value = "refreshed_token"
        
        # Mock security validation
        mock.validate_input.return_value = {"is_safe": True, "sanitized": "safe_input"}
        mock.check_security_policy.return_value = {"compliant": True, "violations": []}
        
        return mock

    @pytest.fixture
    def mock_audit_logger_service(self):
        """Mock audit logger service for testing."""
        mock = AsyncMock()
        
        # Mock action logging
        mock.log_action.return_value = {"log_id": "test-log-123"}
        mock.log_user_action.return_value = {"log_id": "test-user-log-123"}
        mock.log_system_action.return_value = {"log_id": "test-system-log-123"}
        
        # Mock audit trail
        mock.get_audit_trail.return_value = [
            {"action": "create", "user": "test-user", "timestamp": datetime.now()},
            {"action": "update", "user": "test-user", "timestamp": datetime.now()}
        ]
        
        mock.search_audit_log.return_value = [
            {"action": "delete", "user": "admin", "timestamp": datetime.now()}
        ]
        
        # Mock audit export
        mock.export_audit_log.return_value = {"export_id": "test-audit-export-123"}
        mock.get_audit_summary.return_value = {
            "total_actions": 1000,
            "unique_users": 50,
            "action_types": ["create", "update", "delete"]
        }
        
        return mock


# Service factory for creating mock services
@pytest.fixture
def mock_service_factory():
    """Factory for creating mock services with custom configurations."""
    
    def create_mock_ad_service(**kwargs):
        """Create a mock ad service with custom configurations."""
        mock = AsyncMock(spec=AdService)
        
        # Set default values
        mock.create_ad.return_value = kwargs.get("create_result", Mock())
        mock.get_ad.return_value = kwargs.get("get_result", Mock())
        mock.update_ad.return_value = kwargs.get("update_result", Mock())
        mock.delete_ad.return_value = kwargs.get("delete_result", True)
        mock.list_ads.return_value = kwargs.get("list_result", [Mock(), Mock(), Mock()])
        
        return mock
    
    def create_mock_campaign_service(**kwargs):
        """Create a mock campaign service with custom configurations."""
        mock = AsyncMock(spec=CampaignService)
        
        # Set default values
        mock.create_campaign.return_value = kwargs.get("create_result", Mock())
        mock.get_campaign.return_value = kwargs.get("get_result", Mock())
        mock.update_campaign.return_value = kwargs.get("update_result", Mock())
        mock.delete_campaign.return_value = kwargs.get("delete_result", True)
        mock.list_campaigns.return_value = kwargs.get("list_result", [Mock(), Mock(), Mock()])
        
        return mock
    
    def create_mock_use_case(**kwargs):
        """Create a mock use case with custom configurations."""
        mock = AsyncMock()
        
        # Set default values
        mock.execute.return_value = kwargs.get("execute_result", Mock())
        
        return mock
    
    return {
        "create_mock_ad_service": create_mock_ad_service,
        "create_mock_campaign_service": create_mock_campaign_service,
        "create_mock_use_case": create_mock_use_case
    }


# Service scenario fixtures
@pytest.fixture
def service_scenario_fixtures():
    """Fixtures for different service testing scenarios."""
    
    def get_happy_path_services():
        """Get services configured for happy path testing."""
        return {
            "ad_service": AsyncMock(create_ad=AsyncMock(return_value=Mock(id="happy-ad-123"))),
            "campaign_service": AsyncMock(create_campaign=AsyncMock(return_value=Mock(id="happy-campaign-123"))),
            "optimization_service": AsyncMock(optimize_ad=AsyncMock(return_value={"improvements": {"ctr": 0.15}}))
        }
    
    def get_error_scenario_services():
        """Get services configured for error scenario testing."""
        return {
            "ad_service": AsyncMock(create_ad=AsyncMock(side_effect=Exception("Database error"))),
            "campaign_service": AsyncMock(create_campaign=AsyncMock(side_effect=ValueError("Invalid data"))),
            "optimization_service": AsyncMock(optimize_ad=AsyncMock(side_effect=RuntimeError("Service unavailable")))
        }
    
    def get_performance_test_services():
        """Get services configured for performance testing."""
        return {
            "ad_service": AsyncMock(create_ad=AsyncMock(return_value=Mock(id="perf-ad-123"))),
            "campaign_service": AsyncMock(create_campaign=AsyncMock(return_value=Mock(id="perf-campaign-123"))),
            "cache_service": AsyncMock(get=AsyncMock(return_value="cached_data"))
        }
    
    return {
        "get_happy_path_services": get_happy_path_services,
        "get_error_scenario_services": get_error_scenario_services,
        "get_performance_test_services": get_performance_test_services
    }

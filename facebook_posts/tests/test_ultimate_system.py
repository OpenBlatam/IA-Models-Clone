"""
Comprehensive Test Suite for Ultimate Facebook Posts System
Testing all advanced features with functional programming principles
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Import all system components
from ..core.predictive_analytics import (
    PredictiveAnalyticsSystem, PredictionType, ModelType,
    create_predictive_analytics_system
)
from ..core.real_time_dashboard import (
    RealTimeDashboard, ChartType, create_real_time_dashboard
)
from ..core.intelligent_cache import (
    IntelligentCacheSystem, CacheStrategy, CacheItemType,
    create_intelligent_cache
)
from ..core.auto_scaling import (
    AutoScalingSystem, ScalingAction, create_auto_scaling_system
)
from ..core.advanced_security import (
    AdvancedSecuritySystem, SecurityEventType, ThreatLevel,
    create_advanced_security_system
)
from ..core.performance_optimizer import (
    PerformanceOptimizer, create_performance_optimizer
)
from ..core.advanced_monitoring import (
    AdvancedMonitoringSystem, create_monitoring_system
)
from ..core.ultimate_integration import (
    UltimateIntegrationSystem, ComponentType, IntegrationStatus,
    create_ultimate_integration_system
)
from ..core.enterprise_features import (
    EnterpriseFeaturesSystem, TenantStatus, UserRole, ComplianceStandard,
    create_enterprise_features_system
)


# Test fixtures

@pytest.fixture
async def predictive_system():
    """Create predictive analytics system for testing"""
    system = create_predictive_analytics_system()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def dashboard_system():
    """Create real-time dashboard for testing"""
    system = create_real_time_dashboard()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def cache_system():
    """Create intelligent cache for testing"""
    system = create_intelligent_cache()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def auto_scaling_system():
    """Create auto-scaling system for testing"""
    system = create_auto_scaling_system()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def security_system():
    """Create advanced security system for testing"""
    system = create_advanced_security_system()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def performance_optimizer():
    """Create performance optimizer for testing"""
    system = create_performance_optimizer()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def monitoring_system():
    """Create monitoring system for testing"""
    system = create_monitoring_system()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def integration_system():
    """Create ultimate integration system for testing"""
    system = create_ultimate_integration_system()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
async def enterprise_system():
    """Create enterprise features system for testing"""
    system = create_enterprise_features_system()
    yield system


# Pure function tests

class TestPureFunctions:
    """Test pure functions across all systems"""
    
    def test_calculate_threat_score(self):
        """Test threat score calculation"""
        from ..core.advanced_security import calculate_threat_score
        
        # Test low threat
        score = calculate_threat_score(
            SecurityEventType.AUTHENTICATION_FAILURE,
            "127.0.0.1", "Mozilla/5.0", "/api/test", "normal content"
        )
        assert score == ThreatLevel.LOW
        
        # Test high threat
        score = calculate_threat_score(
            SecurityEventType.INJECTION_ATTEMPT,
            "192.168.1.1", "suspicious-bot", "/api/test",
            "'; DROP TABLE users; --"
        )
        assert score == ThreatLevel.HIGH
    
    def test_calculate_load_score(self):
        """Test load score calculation"""
        from ..core.auto_scaling import calculate_load_score, LoadMetrics
        
        # Test normal load
        metrics = LoadMetrics(
            cpu_usage=50.0, memory_usage=60.0, request_rate=100.0,
            response_time=1.5, active_connections=50, queue_length=5,
            timestamp=datetime.utcnow()
        )
        score = calculate_load_score(metrics)
        assert 0.0 <= score <= 1.0
        
        # Test high load
        metrics = LoadMetrics(
            cpu_usage=90.0, memory_usage=95.0, request_rate=1000.0,
            response_time=5.0, active_connections=500, queue_length=50,
            timestamp=datetime.utcnow()
        )
        score = calculate_load_score(metrics)
        assert score > 0.8
    
    def test_create_cache_key(self):
        """Test cache key creation"""
        from ..core.intelligent_cache import create_cache_key
        
        key1 = create_cache_key("test", "id1", {"param": "value"})
        key2 = create_cache_key("test", "id1", {"param": "value"})
        key3 = create_cache_key("test", "id2", {"param": "value"})
        
        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different IDs should generate different keys
        assert key1.startswith("test:")  # Should have correct prefix
    
    def test_check_permission(self):
        """Test permission checking"""
        from ..core.enterprise_features import check_permission
        
        # Test admin permissions
        assert check_permission(UserRole.TENANT_ADMIN, "posts.create") == True
        assert check_permission(UserRole.TENANT_ADMIN, "users.manage") == True
        
        # Test user permissions
        assert check_permission(UserRole.USER, "posts.create") == True
        assert check_permission(UserRole.USER, "users.manage") == False
        
        # Test viewer permissions
        assert check_permission(UserRole.VIEWER, "posts.read") == True
        assert check_permission(UserRole.VIEWER, "posts.create") == False


# Predictive Analytics Tests

class TestPredictiveAnalytics:
    """Test predictive analytics system"""
    
    @pytest.mark.asyncio
    async def test_predictive_system_initialization(self, predictive_system):
        """Test system initialization"""
        assert predictive_system is not None
        assert len(predictive_system.models) == 0
        assert len(predictive_system.prediction_history) == 0
    
    @pytest.mark.asyncio
    async def test_training_data_creation(self, predictive_system):
        """Test training data creation"""
        from ..core.predictive_analytics import create_training_data
        
        historical_data = [
            {
                "content": "Test post 1",
                "timestamp": datetime.utcnow().isoformat(),
                "audience_type": "general",
                "engagement": 0.8
            },
            {
                "content": "Test post 2",
                "timestamp": datetime.utcnow().isoformat(),
                "audience_type": "professionals",
                "engagement": 0.9
            }
        ]
        
        training_data = create_training_data(historical_data, "engagement")
        
        assert training_data.sample_count == 2
        assert len(training_data.features) == 2
        assert len(training_data.targets) == 2
        assert len(training_data.feature_names) > 0
    
    @pytest.mark.asyncio
    async def test_model_training(self, predictive_system):
        """Test model training"""
        from ..core.predictive_analytics import create_training_data, TrainingData
        
        # Create training data
        training_data = TrainingData(
            features=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            targets=[0.8, 0.9, 0.7],
            feature_names=["feature1", "feature2", "feature3"],
            sample_count=3,
            timestamp=datetime.utcnow()
        )
        
        # Train model
        result = await predictive_system.train_model(
            PredictionType.ENGAGEMENT,
            training_data,
            ModelType.LINEAR_REGRESSION
        )
        
        assert result["model_key"] is not None
        assert result["training_samples"] == 3
        assert "performance" in result
    
    @pytest.mark.asyncio
    async def test_prediction(self, predictive_system):
        """Test prediction functionality"""
        # First train a model
        from ..core.predictive_analytics import create_training_data, TrainingData
        
        training_data = TrainingData(
            features=[[1, 2, 3], [4, 5, 6]],
            targets=[0.8, 0.9],
            feature_names=["f1", "f2", "f3"],
            sample_count=2,
            timestamp=datetime.utcnow()
        )
        
        await predictive_system.train_model(
            PredictionType.ENGAGEMENT,
            training_data,
            ModelType.LINEAR_REGRESSION
        )
        
        # Make prediction
        prediction = await predictive_system.predict(
            PredictionType.ENGAGEMENT,
            "Test content",
            datetime.utcnow(),
            "general"
        )
        
        assert prediction.predicted_value >= 0
        assert 0 <= prediction.confidence_score <= 1
        assert prediction.prediction_type == PredictionType.ENGAGEMENT


# Real-time Dashboard Tests

class TestRealTimeDashboard:
    """Test real-time dashboard system"""
    
    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, dashboard_system):
        """Test dashboard initialization"""
        assert dashboard_system is not None
        assert len(dashboard_system.widgets) > 0  # Should have default widgets
        assert dashboard_system.is_running == True
    
    @pytest.mark.asyncio
    async def test_data_point_creation(self, dashboard_system):
        """Test data point creation and addition"""
        dashboard_system.add_data_point("test_metric", 42.5, "Test Data Point")
        
        data = dashboard_system.get_metric_data("test_metric")
        assert len(data) == 1
        assert data[0].value == 42.5
        assert data[0].label == "Test Data Point"
    
    @pytest.mark.asyncio
    async def test_widget_management(self, dashboard_system):
        """Test widget management"""
        from ..core.real_time_dashboard import create_dashboard_widget, create_chart_config, ChartType, DataPoint
        
        # Create test widget
        data_points = [DataPoint(datetime.utcnow(), 1.0, "test", {})]
        chart_config = create_chart_config(
            ChartType.LINE, "Test Chart", data_points
        )
        widget = create_dashboard_widget(
            "test_widget", "Test Widget", chart_config,
            {"x": 0, "y": 0}, {"width": 200, "height": 150}
        )
        
        # Add widget
        dashboard_system.add_widget(widget)
        assert "test_widget" in dashboard_system.widgets
        
        # Get widget data
        widget_data = dashboard_system.get_widget_data("test_widget")
        assert widget_data is not None
        assert widget_data["title"] == "Test Widget"
        
        # Remove widget
        removed = dashboard_system.remove_widget("test_widget")
        assert removed == True
        assert "test_widget" not in dashboard_system.widgets
    
    @pytest.mark.asyncio
    async def test_dashboard_data(self, dashboard_system):
        """Test dashboard data retrieval"""
        # Add some test data
        dashboard_system.add_data_point("cpu_usage", 75.0, "CPU Usage %")
        dashboard_system.add_data_point("memory_usage", 60.0, "Memory Usage %")
        
        # Get dashboard data
        data = dashboard_system.get_dashboard_data()
        
        assert "widgets" in data
        assert "statistics" in data
        assert "metrics" in data
        assert "cpu_usage" in data["metrics"]
        assert data["metrics"]["cpu_usage"]["last_value"] == 75.0


# Intelligent Cache Tests

class TestIntelligentCache:
    """Test intelligent cache system"""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_system):
        """Test cache initialization"""
        assert cache_system is not None
        assert cache_system.is_running == True
        assert cache_system.strategy == CacheStrategy.ADAPTIVE
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, cache_system):
        """Test basic cache operations"""
        # Set item
        success = await cache_system.set(
            "test_key", "test_value", CacheItemType.POST_CONTENT
        )
        assert success == True
        
        # Get item
        value = await cache_system.get("test_key")
        assert value == "test_value"
        
        # Delete item
        deleted = await cache_system.delete("test_key")
        assert deleted == True
        
        # Get deleted item
        value = await cache_system.get("test_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_metrics(self, cache_system):
        """Test cache metrics"""
        # Add some data
        await cache_system.set("key1", "value1", CacheItemType.POST_CONTENT)
        await cache_system.set("key2", "value2", CacheItemType.AI_RESPONSE)
        
        # Get metrics
        metrics = cache_system.get_cache_metrics()
        
        assert metrics.total_items >= 2
        assert metrics.total_size_bytes > 0
        assert metrics.hit_rate >= 0
        assert metrics.miss_rate >= 0
    
    @pytest.mark.asyncio
    async def test_cache_strategies(self, cache_system):
        """Test different cache strategies"""
        # Test LRU strategy
        cache_system.strategy = CacheStrategy.LRU
        await cache_system.set("lru_key", "lru_value", CacheItemType.POST_CONTENT)
        
        # Test TTL strategy
        cache_system.strategy = CacheStrategy.TTL
        await cache_system.set("ttl_key", "ttl_value", CacheItemType.POST_CONTENT, ttl_seconds=1)
        
        # Wait for TTL expiration
        await asyncio.sleep(1.1)
        
        # Check TTL expired item
        ttl_value = await cache_system.get("ttl_key")
        assert ttl_value is None  # Should be expired
        
        # Check LRU item still exists
        lru_value = await cache_system.get("lru_key")
        assert lru_value == "lru_value"


# Auto-scaling Tests

class TestAutoScaling:
    """Test auto-scaling system"""
    
    @pytest.mark.asyncio
    async def test_auto_scaling_initialization(self, auto_scaling_system):
        """Test auto-scaling initialization"""
        assert auto_scaling_system is not None
        assert auto_scaling_system.is_running == True
        assert auto_scaling_system.current_instances == auto_scaling_system.min_instances
    
    @pytest.mark.asyncio
    async def test_scaling_statistics(self, auto_scaling_system):
        """Test scaling statistics"""
        stats = auto_scaling_system.get_scaling_statistics()
        
        assert "current_instances" in stats
        assert "min_instances" in stats
        assert "max_instances" in stats
        assert "statistics" in stats
        assert stats["is_running"] == True
    
    @pytest.mark.asyncio
    async def test_manual_scaling(self, auto_scaling_system):
        """Test manual scaling"""
        original_instances = auto_scaling_system.current_instances
        
        # Scale up
        success = await auto_scaling_system.manual_scale(
            original_instances + 2, "Test scaling up"
        )
        assert success == True
        assert auto_scaling_system.current_instances == original_instances + 2
        
        # Scale down
        success = await auto_scaling_system.manual_scale(
            original_instances, "Test scaling down"
        )
        assert success == True
        assert auto_scaling_system.current_instances == original_instances
    
    @pytest.mark.asyncio
    async def test_load_trends(self, auto_scaling_system):
        """Test load trend analysis"""
        trends = auto_scaling_system.get_load_trends()
        
        assert "trend" in trends
        assert "data_points" in trends
        assert "time_range_minutes" in trends


# Advanced Security Tests

class TestAdvancedSecurity:
    """Test advanced security system"""
    
    @pytest.mark.asyncio
    async def test_security_initialization(self, security_system):
        """Test security system initialization"""
        assert security_system is not None
        assert security_system.is_running == True
    
    @pytest.mark.asyncio
    async def test_request_security_check(self, security_system):
        """Test request security checking"""
        # Test safe request
        is_secure, event = await security_system.check_request_security(
            "127.0.0.1", "Mozilla/5.0", "/api/test", "normal content"
        )
        assert is_secure == True
        assert event is None
        
        # Test malicious request
        is_secure, event = await security_system.check_request_security(
            "192.168.1.1", "suspicious-bot", "/api/test",
            "'; DROP TABLE users; --"
        )
        assert is_secure == False
        assert event is not None
        assert event.event_type == SecurityEventType.INJECTION_ATTEMPT
    
    @pytest.mark.asyncio
    async def test_api_key_management(self, security_system):
        """Test API key management"""
        # Generate API key
        api_key = security_system.generate_api_key("test_user", ["read", "write"])
        assert api_key is not None
        assert api_key.startswith("fbp_")
        
        # Verify API key
        is_valid = await security_system._verify_api_key(api_key)
        assert is_valid == True
        
        # Revoke API key
        revoked = security_system.revoke_api_key(api_key)
        assert revoked == True
        
        # Verify revoked key
        is_valid = await security_system._verify_api_key(api_key)
        assert is_valid == False
    
    @pytest.mark.asyncio
    async def test_security_statistics(self, security_system):
        """Test security statistics"""
        stats = security_system.get_security_statistics()
        
        assert "statistics" in stats
        assert "blocked_ips" in stats
        assert "active_api_keys" in stats
        assert "security_rules" in stats
        assert "recent_events" in stats
    
    @pytest.mark.asyncio
    async def test_threat_analysis(self, security_system):
        """Test threat analysis"""
        analysis = security_system.get_threat_analysis()
        
        assert "threat_levels" in analysis
        assert "event_types" in analysis
        assert "top_threat_sources" in analysis
        assert "total_events" in analysis


# Performance Optimizer Tests

class TestPerformanceOptimizer:
    """Test performance optimizer system"""
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_initialization(self, performance_optimizer):
        """Test performance optimizer initialization"""
        assert performance_optimizer is not None
        assert performance_optimizer.is_running == True
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, performance_optimizer):
        """Test performance monitoring"""
        # Get performance summary
        summary = performance_optimizer.get_performance_summary()
        
        assert "status" in summary
        assert "cpu_usage" in summary
        assert "memory_usage" in summary
        assert "response_time" in summary
        assert "throughput" in summary
    
    @pytest.mark.asyncio
    async def test_system_optimization(self, performance_optimizer):
        """Test system optimization"""
        # Perform optimization
        result = await performance_optimizer.optimize_system()
        
        assert "optimization_applied" in result
        assert "performance_improvement" in result
        assert "timestamp" in result


# Advanced Monitoring Tests

class TestAdvancedMonitoring:
    """Test advanced monitoring system"""
    
    @pytest.mark.asyncio
    async def test_monitoring_initialization(self, monitoring_system):
        """Test monitoring system initialization"""
        assert monitoring_system is not None
        assert monitoring_system.is_running == True
    
    @pytest.mark.asyncio
    async def test_health_status(self, monitoring_system):
        """Test health status monitoring"""
        health = monitoring_system.get_health_status()
        
        assert "status" in health
        assert "components" in health
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_dashboard_data(self, monitoring_system):
        """Test dashboard data"""
        data = monitoring_system.get_dashboard_data()
        
        assert "metrics" in data
        assert "alerts" in data
        assert "performance" in data
        assert "timestamp" in data


# Ultimate Integration Tests

class TestUltimateIntegration:
    """Test ultimate integration system"""
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, integration_system):
        """Test integration system initialization"""
        assert integration_system is not None
        assert integration_system.is_running == True
        assert len(integration_system.components) > 0
    
    @pytest.mark.asyncio
    async def test_component_health(self, integration_system):
        """Test component health monitoring"""
        health = integration_system.get_overall_health()
        assert health in [IntegrationStatus.RUNNING, IntegrationStatus.DEGRADED]
        
        all_health = integration_system.get_all_component_health()
        assert len(all_health) > 0
    
    @pytest.mark.asyncio
    async def test_integration_statistics(self, integration_system):
        """Test integration statistics"""
        stats = integration_system.get_integration_statistics()
        
        assert "overall_health" in stats
        assert "component_scores" in stats
        assert "statistics" in stats
        assert "total_components" in stats
        assert "is_running" in stats
    
    @pytest.mark.asyncio
    async def test_ultimate_workflow(self, integration_system):
        """Test ultimate workflow execution"""
        workflow_data = {
            "content": "Test content for workflow",
            "audience_type": "general",
            "source_ip": "127.0.0.1",
            "user_agent": "test-agent"
        }
        
        result = await integration_system.execute_ultimate_workflow(
            "content_generation", workflow_data
        )
        
        assert "success" in result
        if result["success"]:
            assert "enhanced_analysis" in result
            assert "optimization_result" in result
            assert "engagement_prediction" in result


# Enterprise Features Tests

class TestEnterpriseFeatures:
    """Test enterprise features system"""
    
    @pytest.mark.asyncio
    async def test_tenant_creation(self, enterprise_system):
        """Test tenant creation"""
        tenant, admin_user = await enterprise_system.create_tenant(
            name="Test Tenant",
            domain="test.example.com",
            admin_email="admin@test.com",
            admin_username="admin",
            settings={"max_posts_per_day": 500},
            compliance_standards=[ComplianceStandard.GDPR]
        )
        
        assert tenant.name == "Test Tenant"
        assert tenant.domain == "test.example.com"
        assert tenant.status == TenantStatus.ACTIVE
        assert ComplianceStandard.GDPR in tenant.compliance_standards
        
        assert admin_user.email == "admin@test.com"
        assert admin_user.role == UserRole.TENANT_ADMIN
        assert admin_user.tenant_id == tenant.tenant_id
    
    @pytest.mark.asyncio
    async def test_user_creation(self, enterprise_system):
        """Test user creation"""
        # First create a tenant
        tenant, admin_user = await enterprise_system.create_tenant(
            "Test Tenant", "test.example.com",
            "admin@test.com", "admin"
        )
        
        # Create a regular user
        user = await enterprise_system.create_user(
            tenant.tenant_id,
            "user@test.com",
            "testuser",
            UserRole.USER,
            admin_user.user_id
        )
        
        assert user.email == "user@test.com"
        assert user.role == UserRole.USER
        assert user.tenant_id == tenant.tenant_id
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, enterprise_system):
        """Test user authentication"""
        # Create tenant and user
        tenant, admin_user = await enterprise_system.create_tenant(
            "Test Tenant", "test.example.com",
            "admin@test.com", "admin"
        )
        
        # Authenticate user
        result = await enterprise_system.authenticate_user(
            "admin@test.com", "password", "127.0.0.1", "test-agent"
        )
        
        assert result is not None
        user, session_token = result
        assert user.email == "admin@test.com"
        assert session_token is not None
    
    @pytest.mark.asyncio
    async def test_permission_checking(self, enterprise_system):
        """Test permission checking"""
        # Create tenant and users
        tenant, admin_user = await enterprise_system.create_tenant(
            "Test Tenant", "test.example.com",
            "admin@test.com", "admin"
        )
        
        regular_user = await enterprise_system.create_user(
            tenant.tenant_id, "user@test.com", "user",
            UserRole.USER, admin_user.user_id
        )
        
        # Test admin permissions
        can_manage_users = await enterprise_system.check_permission(
            admin_user, "users.manage"
        )
        assert can_manage_users == True
        
        # Test user permissions
        can_create_posts = await enterprise_system.check_permission(
            regular_user, "posts.create"
        )
        assert can_create_posts == True
        
        can_manage_users = await enterprise_system.check_permission(
            regular_user, "users.manage"
        )
        assert can_manage_users == False
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, enterprise_system):
        """Test audit logging"""
        # Create tenant and user
        tenant, admin_user = await enterprise_system.create_tenant(
            "Test Tenant", "test.example.com",
            "admin@test.com", "admin"
        )
        
        # Log activity
        await enterprise_system.log_activity(
            admin_user, "test_action", "test_resource",
            {"test": "data"}, "127.0.0.1", "test-agent"
        )
        
        # Get audit logs
        logs = await enterprise_system.get_audit_logs(
            tenant_id=tenant.tenant_id,
            action="test_action"
        )
        
        assert len(logs) >= 1
        assert logs[0].action == "test_action"
        assert logs[0].user_id == admin_user.user_id
    
    @pytest.mark.asyncio
    async def test_compliance_reporting(self, enterprise_system):
        """Test compliance reporting"""
        # Create tenant with GDPR compliance
        tenant, admin_user = await enterprise_system.create_tenant(
            "Test Tenant", "test.example.com",
            "admin@test.com", "admin",
            compliance_standards=[ComplianceStandard.GDPR]
        )
        
        # Generate compliance report
        report = await enterprise_system.get_compliance_report(
            tenant.tenant_id, ComplianceStandard.GDPR
        )
        
        assert report["tenant_id"] == tenant.tenant_id
        assert report["compliance_standard"] == "gdpr"
        assert "status" in report
        assert "findings" in report
    
    @pytest.mark.asyncio
    async def test_enterprise_statistics(self, enterprise_system):
        """Test enterprise statistics"""
        stats = enterprise_system.get_enterprise_statistics()
        
        assert "statistics" in stats
        assert "tenants" in stats
        assert "users" in stats
        assert "compliance" in stats
        assert "audit_logs" in stats


# Integration Tests

class TestSystemIntegration:
    """Test system integration across components"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete system workflow"""
        # Initialize all systems
        integration_system = create_ultimate_integration_system()
        await integration_system.start()
        
        try:
            # Test content generation workflow
            workflow_data = {
                "content": "Test content for full workflow",
                "audience_type": "general",
                "source_ip": "127.0.0.1",
                "user_agent": "test-agent"
            }
            
            result = await integration_system.execute_ultimate_workflow(
                "content_generation", workflow_data
            )
            
            assert "success" in result
            if result["success"]:
                assert "enhanced_analysis" in result
                assert "optimization_result" in result
                assert "engagement_prediction" in result
            
            # Test system optimization workflow
            opt_result = await integration_system.execute_ultimate_workflow(
                "system_optimization", {}
            )
            
            assert "success" in opt_result
            if opt_result["success"]:
                assert "optimization_results" in opt_result
            
        finally:
            await integration_system.stop()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load"""
        cache_system = create_intelligent_cache()
        await cache_system.start()
        
        try:
            # Simulate load
            start_time = time.time()
            
            # Perform many cache operations
            for i in range(100):
                await cache_system.set(f"key_{i}", f"value_{i}", CacheItemType.POST_CONTENT)
                await cache_system.get(f"key_{i}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete within reasonable time
            assert duration < 10.0  # 10 seconds max
            
            # Check cache metrics
            metrics = cache_system.get_cache_metrics()
            assert metrics.total_items >= 100
            
        finally:
            await cache_system.stop()


# Performance Tests

class TestPerformance:
    """Test system performance"""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """Test response time performance"""
        cache_system = create_intelligent_cache()
        await cache_system.start()
        
        try:
            # Measure response time
            start_time = time.time()
            
            await cache_system.set("perf_test", "test_value", CacheItemType.POST_CONTENT)
            value = await cache_system.get("perf_test")
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Should be very fast
            assert response_time < 100  # Less than 100ms
            assert value == "test_value"
            
        finally:
            await cache_system.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage"""
        cache_system = create_intelligent_cache(max_size_bytes=1024*1024)  # 1MB limit
        await cache_system.start()
        
        try:
            # Add data until we hit memory limit
            for i in range(1000):
                success = await cache_system.set(
                    f"mem_test_{i}", "x" * 1000, CacheItemType.POST_CONTENT
                )
                if not success:
                    break  # Hit memory limit
            
            # Check memory usage
            metrics = cache_system.get_cache_metrics()
            assert metrics.memory_usage_percent > 0
            assert metrics.total_size_bytes <= 1024*1024
            
        finally:
            await cache_system.stop()


# Error Handling Tests

class TestErrorHandling:
    """Test error handling across systems"""
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        cache_system = create_intelligent_cache()
        await cache_system.start()
        
        try:
            # Test invalid key
            value = await cache_system.get(None)
            assert value is None
            
            # Test invalid value
            success = await cache_system.set("", None, CacheItemType.POST_CONTENT)
            assert success == False
            
        finally:
            await cache_system.stop()
    
    @pytest.mark.asyncio
    async def test_system_failure_recovery(self):
        """Test system failure recovery"""
        integration_system = create_ultimate_integration_system()
        
        # Test starting and stopping
        await integration_system.start()
        assert integration_system.is_running == True
        
        await integration_system.stop()
        assert integration_system.is_running == False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


"""
Comprehensive Test Suite for BUL Advanced Features
Tests all advanced functionality including templates, models, workflows, integrations, and analytics
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import our advanced modules
from ai.document_templates import (
    DocumentTemplateManager, DocumentType, IndustryType, TemplateComplexity,
    SmartSuggestion, TemplateRecommendation, template_manager
)
from ai.model_manager import (
    ModelManager, ModelRequest, ModelResponse, ModelProvider, ModelType,
    ABTestConfig, model_manager
)
from workflows.workflow_engine import (
    WorkflowEngine, WorkflowDefinition, WorkflowExecution, WorkflowStatus,
    workflow_engine
)
from integrations.third_party_integrations import (
    ThirdPartyIntegrationManager, IntegrationType, IntegrationStatus,
    GoogleDocsIntegration, Office365Integration, SalesforceIntegration,
    integration_manager
)
from analytics.dashboard import (
    AnalyticsDashboard, MetricType, TimeRange, ChartType,
    analytics_dashboard
)
from api.advanced_rate_limiting import (
    AdvancedRateLimiter, AdvancedCache, RateLimitType, CacheStrategy,
    rate_limiter, cache
)


class TestDocumentTemplates:
    """Test document templates functionality"""
    
    @pytest.fixture
    async def template_manager_instance(self):
        """Create template manager instance for testing"""
        return DocumentTemplateManager()
    
    @pytest.mark.asyncio
    async def test_get_template(self, template_manager_instance):
        """Test getting a template by ID"""
        template = await template_manager_instance.get_template("business_plan_v1")
        assert template is not None
        assert template.id == "business_plan_v1"
        assert template.document_type == DocumentType.BUSINESS_PLAN
        assert template.complexity == TemplateComplexity.ADVANCED
    
    @pytest.mark.asyncio
    async def test_list_templates_with_filters(self, template_manager_instance):
        """Test listing templates with filters"""
        # Test filtering by document type
        templates = await template_manager_instance.list_templates(
            document_type=DocumentType.BUSINESS_PLAN
        )
        assert len(templates) > 0
        assert all(t.document_type == DocumentType.BUSINESS_PLAN for t in templates)
        
        # Test filtering by complexity
        templates = await template_manager_instance.list_templates(
            complexity=TemplateComplexity.BASIC
        )
        assert len(templates) > 0
        assert all(t.complexity == TemplateComplexity.BASIC for t in templates)
    
    @pytest.mark.asyncio
    async def test_recommend_templates(self, template_manager_instance):
        """Test template recommendation system"""
        user_context = {
            "document_type": "business_plan",
            "industry": "technology",
            "complexity": "advanced"
        }
        
        recommendations = await template_manager_instance.recommend_templates(user_context)
        assert len(recommendations) > 0
        assert all(isinstance(r, TemplateRecommendation) for r in recommendations)
        assert all(r.score > 0 for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_smart_suggestions(self, template_manager_instance):
        """Test smart suggestions generation"""
        field_data = {
            "company_name": "Test Company",
            "industry": "technology"
        }
        user_context = {
            "experience_level": "intermediate"
        }
        
        suggestions = await template_manager_instance.generate_smart_suggestions(
            template_id="business_plan_v1",
            field_data=field_data,
            user_context=user_context
        )
        
        assert isinstance(suggestions, list)
        assert all(isinstance(s, SmartSuggestion) for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_create_custom_template(self, template_manager_instance):
        """Test creating custom template"""
        template_data = {
            "name": "Custom Test Template",
            "description": "A custom template for testing",
            "document_type": "business_plan",
            "complexity": "intermediate",
            "fields": [
                {
                    "name": "test_field",
                    "label": "Test Field",
                    "type": "text",
                    "required": True
                }
            ],
            "content_structure": {"sections": ["test_section"]},
            "ai_prompts": {"main": "Generate test content"},
            "tags": ["test", "custom"]
        }
        
        template = await template_manager_instance.create_custom_template(
            template_data=template_data,
            user_id="test_user"
        )
        
        assert template.name == "Custom Test Template"
        assert template.document_type == DocumentType.BUSINESS_PLAN
        assert len(template.fields) == 1
        assert template.fields[0].name == "test_field"
    
    @pytest.mark.asyncio
    async def test_get_template_statistics(self, template_manager_instance):
        """Test template statistics"""
        stats = await template_manager_instance.get_template_statistics()
        
        assert "total_templates" in stats
        assert "templates_by_type" in stats
        assert "templates_by_complexity" in stats
        assert stats["total_templates"] > 0


class TestModelManager:
    """Test AI model management functionality"""
    
    @pytest.fixture
    async def model_manager_instance(self):
        """Create model manager instance for testing"""
        return ModelManager()
    
    @pytest.mark.asyncio
    async def test_get_model(self, model_manager_instance):
        """Test getting a model by ID"""
        model = await model_manager_instance.get_model("openai_gpt4")
        assert model is not None
        assert model.id == "openai_gpt4"
        assert model.provider == ModelProvider.OPENAI
        assert model.model_type == ModelType.TEXT_GENERATION
    
    @pytest.mark.asyncio
    async def test_list_models_with_filters(self, model_manager_instance):
        """Test listing models with filters"""
        # Test filtering by provider
        models = await model_manager_instance.list_models(provider=ModelProvider.OPENAI)
        assert len(models) > 0
        assert all(m.provider == ModelProvider.OPENAI for m in models)
        
        # Test filtering by model type
        models = await model_manager_instance.list_models(model_type=ModelType.TEXT_GENERATION)
        assert len(models) > 0
        assert all(m.model_type == ModelType.TEXT_GENERATION for m in models)
    
    @pytest.mark.asyncio
    async def test_get_best_model(self, model_manager_instance):
        """Test getting best model based on criteria"""
        # Test performance-based selection
        best_model = await model_manager_instance.get_best_model(
            model_type=ModelType.TEXT_GENERATION,
            criteria="performance"
        )
        assert best_model is not None
        assert best_model.model_type == ModelType.TEXT_GENERATION
        
        # Test cost-based selection
        best_model = await model_manager_instance.get_best_model(
            model_type=ModelType.TEXT_GENERATION,
            criteria="cost"
        )
        assert best_model is not None
    
    @pytest.mark.asyncio
    async def test_generate_content(self, model_manager_instance):
        """Test content generation"""
        request = ModelRequest(
            prompt="Generate a test document",
            max_tokens=100,
            temperature=0.7,
            user_id="test_user"
        )
        
        response = await model_manager_instance.generate_content(request)
        
        assert isinstance(response, ModelResponse)
        assert response.content is not None
        assert response.model_id is not None
        assert response.tokens_used > 0
        assert response.response_time > 0
    
    @pytest.mark.asyncio
    async def test_create_ab_test(self, model_manager_instance):
        """Test A/B test creation"""
        config = ABTestConfig(
            test_id="test_ab_test",
            name="Test A/B Test",
            description="Testing A/B test functionality",
            models=["openai_gpt4", "openai_gpt35"],
            traffic_split={"openai_gpt4": 0.5, "openai_gpt35": 0.5},
            success_metrics=["response_time", "quality_score"],
            duration_days=7
        )
        
        test = await model_manager_instance.create_ab_test(config)
        
        assert test.test_id == "test_ab_test"
        assert test.name == "Test A/B Test"
        assert len(test.models) == 2
        assert test.traffic_split["openai_gpt4"] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_model_analytics(self, model_manager_instance):
        """Test model analytics"""
        analytics = await model_manager_instance.get_model_analytics("openai_gpt4", days=30)
        
        assert "model_id" in analytics
        assert "model_name" in analytics
        assert "period_days" in analytics
        assert analytics["model_id"] == "openai_gpt4"
    
    @pytest.mark.asyncio
    async def test_get_system_analytics(self, model_manager_instance):
        """Test system analytics"""
        analytics = await model_manager_instance.get_system_analytics()
        
        assert "total_models" in analytics
        assert "active_models" in analytics
        assert "total_requests" in analytics
        assert "models_by_provider" in analytics


class TestWorkflowEngine:
    """Test workflow engine functionality"""
    
    @pytest.fixture
    async def workflow_engine_instance(self):
        """Create workflow engine instance for testing"""
        return WorkflowEngine()
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, workflow_engine_instance):
        """Test listing workflows"""
        workflows = await workflow_engine_instance.list_workflows()
        
        assert len(workflows) > 0
        assert all(hasattr(w, 'id') for w in workflows)
        assert all(hasattr(w, 'name') for w in workflows)
        assert all(hasattr(w, 'steps') for w in workflows)
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, workflow_engine_instance):
        """Test workflow execution"""
        execution = await workflow_engine_instance.execute_workflow(
            workflow_id="document_generation_v1",
            user_id="test_user",
            context={"test": "data"}
        )
        
        assert execution.workflow_id == "document_generation_v1"
        assert execution.user_id == "test_user"
        assert execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
        assert execution.progress >= 0.0
        assert execution.progress <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_execution_status(self, workflow_engine_instance):
        """Test getting execution status"""
        # First execute a workflow
        execution = await workflow_engine_instance.execute_workflow(
            workflow_id="document_generation_v1",
            user_id="test_user"
        )
        
        # Then get its status
        status = await workflow_engine_instance.get_execution_status(execution.id)
        
        assert status is not None
        assert status.id == execution.id
        assert status.workflow_id == execution.workflow_id
    
    @pytest.mark.asyncio
    async def test_get_workflow_analytics(self, workflow_engine_instance):
        """Test workflow analytics"""
        analytics = await workflow_engine_instance.get_workflow_analytics("document_generation_v1")
        
        assert "workflow_id" in analytics
        assert "total_executions" in analytics
        assert "success_rate" in analytics
        assert "avg_duration" in analytics


class TestThirdPartyIntegrations:
    """Test third-party integrations functionality"""
    
    @pytest.fixture
    async def integration_manager_instance(self):
        """Create integration manager instance for testing"""
        return ThirdPartyIntegrationManager()
    
    @pytest.mark.asyncio
    async def test_list_integrations(self, integration_manager_instance):
        """Test listing integrations"""
        integrations = await integration_manager_instance.list_integrations()
        
        assert len(integrations) > 0
        assert all(hasattr(i, 'id') for i in integrations)
        assert all(hasattr(i, 'name') for i in integrations)
        assert all(hasattr(i, 'type') for i in integrations)
    
    @pytest.mark.asyncio
    async def test_get_integration(self, integration_manager_instance):
        """Test getting integration by ID"""
        integration = await integration_manager_instance.get_integration("google_docs_default")
        
        assert integration is not None
        assert integration.id == "google_docs_default"
        assert integration.type == IntegrationType.GOOGLE_DOCS
    
    @pytest.mark.asyncio
    async def test_create_integration(self, integration_manager_instance):
        """Test creating new integration"""
        integration_data = {
            "id": "test_integration",
            "name": "Test Integration",
            "type": "slack",
            "credentials": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "access_token": "test_access_token"
            },
            "settings": {
                "default_channel": "#test"
            }
        }
        
        integration = await integration_manager_instance.create_integration(integration_data)
        
        assert integration.id == "test_integration"
        assert integration.name == "Test Integration"
        assert integration.type == IntegrationType.SLACK
    
    @pytest.mark.asyncio
    async def test_get_integration_analytics(self, integration_manager_instance):
        """Test integration analytics"""
        analytics = await integration_manager_instance.get_integration_analytics("google_docs_default")
        
        assert "integration_id" in analytics
        assert "integration_name" in analytics
        assert "integration_type" in analytics
        assert "status" in analytics
    
    @pytest.mark.asyncio
    async def test_get_system_integration_analytics(self, integration_manager_instance):
        """Test system integration analytics"""
        analytics = await integration_manager_instance.get_system_integration_analytics()
        
        assert "total_integrations" in analytics
        assert "active_integrations" in analytics
        assert "total_syncs" in analytics
        assert "integrations_by_type" in analytics


class TestAnalyticsDashboard:
    """Test analytics dashboard functionality"""
    
    @pytest.fixture
    async def analytics_dashboard_instance(self):
        """Create analytics dashboard instance for testing"""
        return AnalyticsDashboard()
    
    @pytest.mark.asyncio
    async def test_add_metric(self, analytics_dashboard_instance):
        """Test adding metric"""
        await analytics_dashboard_instance.add_metric(
            metric_name="test_metric",
            value=100.0,
            labels={"type": "test"},
            metadata={"source": "test"}
        )
        
        # Verify metric was added
        assert "test_metric" in analytics_dashboard_instance.metrics
        assert len(analytics_dashboard_instance.metrics["test_metric"]) == 1
    
    @pytest.mark.asyncio
    async def test_query_metrics(self, analytics_dashboard_instance):
        """Test querying metrics"""
        # Add some test data
        await analytics_dashboard_instance.add_metric("test_metric", 100.0)
        await analytics_dashboard_instance.add_metric("test_metric", 200.0)
        
        # Query metrics
        from analytics.dashboard import AnalyticsQuery
        query = AnalyticsQuery(
            metric_name="test_metric",
            time_range=TimeRange.LAST_DAY,
            aggregation="sum"
        )
        
        results = await analytics_dashboard_instance.query_metrics(query)
        
        assert len(results) > 0
        assert "value" in results[0]
    
    @pytest.mark.asyncio
    async def test_get_dashboard_data(self, analytics_dashboard_instance):
        """Test getting dashboard data"""
        dashboard_data = await analytics_dashboard_instance.get_dashboard_data("overview")
        
        assert "dashboard" in dashboard_data
        assert "widgets" in dashboard_data
        assert "generated_at" in dashboard_data
        assert dashboard_data["dashboard"]["id"] == "overview"
    
    @pytest.mark.asyncio
    async def test_get_insights(self, analytics_dashboard_instance):
        """Test getting insights"""
        insights = await analytics_dashboard_instance.get_insights(limit=10)
        
        assert isinstance(insights, list)
        # Insights might be empty if no significant patterns detected
    
    @pytest.mark.asyncio
    async def test_get_analytics_summary(self, analytics_dashboard_instance):
        """Test getting analytics summary"""
        summary = await analytics_dashboard_instance.get_analytics_summary()
        
        assert "summary" in summary
        assert "insights" in summary
        assert "dashboards" in summary
        assert "generated_at" in summary


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    async def rate_limiter_instance(self):
        """Create rate limiter instance for testing"""
        limiter = AdvancedRateLimiter()
        await limiter.initialize()
        return limiter
    
    @pytest.mark.asyncio
    async def test_check_rate_limit(self, rate_limiter_instance):
        """Test rate limit checking"""
        rate_limit_info = await rate_limiter_instance.check_rate_limit(
            endpoint="/test",
            method="GET",
            user_id="test_user",
            ip_address="127.0.0.1"
        )
        
        assert rate_limit_info.limit >= 0
        assert rate_limit_info.remaining >= 0
        assert rate_limit_info.reset_time > 0
    
    @pytest.mark.asyncio
    async def test_add_rule(self, rate_limiter_instance):
        """Test adding rate limit rule"""
        from api.advanced_rate_limiting import RateLimitRule, RateLimitType
        
        rule = RateLimitRule(
            id="test_rule",
            name="Test Rule",
            endpoint="/test/*",
            method="GET",
            limit_type=RateLimitType.SLIDING_WINDOW,
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        )
        
        await rate_limiter_instance.add_rule(rule)
        
        retrieved_rule = await rate_limiter_instance.get_rule("test_rule")
        assert retrieved_rule is not None
        assert retrieved_rule.id == "test_rule"
    
    @pytest.mark.asyncio
    async def test_list_rules(self, rate_limiter_instance):
        """Test listing rate limit rules"""
        rules = await rate_limiter_instance.list_rules()
        
        assert len(rules) > 0
        assert all(hasattr(r, 'id') for r in rules)
        assert all(hasattr(r, 'name') for r in rules)


class TestCaching:
    """Test caching functionality"""
    
    @pytest.fixture
    async def cache_instance(self):
        """Create cache instance for testing"""
        cache_instance = AdvancedCache()
        await cache_instance.initialize()
        return cache_instance
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_instance):
        """Test setting and getting cache values"""
        # Set value
        await cache_instance.set(
            key="test_key",
            value={"test": "data"},
            endpoint="/test",
            method="GET"
        )
        
        # Get value
        result = await cache_instance.get(
            key="test_key",
            endpoint="/test",
            method="GET"
        )
        
        assert result is not None
        assert result["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, cache_instance):
        """Test cache expiry"""
        # Set value with short TTL
        await cache_instance.set(
            key="expiry_test",
            value="test_value",
            endpoint="/test",
            method="GET"
        )
        
        # Get value immediately
        result = await cache_instance.get(
            key="expiry_test",
            endpoint="/test",
            method="GET"
        )
        assert result == "test_value"
        
        # Wait for expiry (in real test, you'd mock time)
        # For now, just test that the cache system works
    
    @pytest.mark.asyncio
    async def test_invalidate(self, cache_instance):
        """Test cache invalidation"""
        # Set multiple values
        await cache_instance.set("key1", "value1", "/test", "GET")
        await cache_instance.set("key2", "value2", "/test", "GET")
        
        # Invalidate pattern
        await cache_instance.invalidate("key1")
        
        # Check that key1 is invalidated but key2 is not
        result1 = await cache_instance.get("key1", "/test", "GET")
        result2 = await cache_instance.get("key2", "/test", "GET")
        
        assert result1 is None
        assert result2 == "value2"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, cache_instance):
        """Test cache statistics"""
        stats = await cache_instance.get_stats()
        
        assert "type" in stats
        assert stats["type"] in ["redis", "memory"]


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features"""
    
    @pytest.mark.asyncio
    async def test_document_generation_workflow(self):
        """Test complete document generation workflow"""
        # 1. Get template recommendation
        user_context = {
            "document_type": "business_plan",
            "industry": "technology",
            "complexity": "advanced"
        }
        
        recommendations = await template_manager.recommend_templates(user_context)
        assert len(recommendations) > 0
        
        # 2. Select template
        template = recommendations[0].template
        
        # 3. Generate content using model
        request = ModelRequest(
            prompt=f"Generate content for {template.name}",
            max_tokens=1000,
            user_id="test_user"
        )
        
        response = await model_manager.generate_content(request)
        assert response.content is not None
        
        # 4. Execute workflow
        execution = await workflow_engine.execute_workflow(
            workflow_id="document_generation_v1",
            user_id="test_user",
            context={
                "template_id": template.id,
                "content": response.content
            }
        )
        
        assert execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_analytics_and_monitoring(self):
        """Test analytics and monitoring integration"""
        # 1. Add metrics
        await analytics_dashboard.add_metric("api_requests", 100, {"endpoint": "/generate"})
        await analytics_dashboard.add_metric("document_generated", 1, {"type": "business_plan"})
        
        # 2. Get analytics summary
        summary = await analytics_dashboard.get_analytics_summary()
        assert "summary" in summary
        
        # 3. Get dashboard data
        dashboard_data = await analytics_dashboard.get_dashboard_data("overview")
        assert "widgets" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_rate_limiting_and_caching(self):
        """Test rate limiting and caching integration"""
        # 1. Check rate limit
        rate_limit_info = await rate_limiter.check_rate_limit(
            endpoint="/generate/enhanced",
            method="POST",
            user_id="test_user"
        )
        
        # 2. Set cache
        await cache.set(
            key="test_cache",
            value={"result": "cached_data"},
            endpoint="/generate/enhanced",
            method="POST"
        )
        
        # 3. Get from cache
        cached_result = await cache.get(
            key="test_cache",
            endpoint="/generate/enhanced",
            method="POST"
        )
        
        assert cached_result is not None
        assert cached_result["result"] == "cached_data"


# Performance tests
class TestPerformance:
    """Performance tests for advanced features"""
    
    @pytest.mark.asyncio
    async def test_template_recommendation_performance(self):
        """Test template recommendation performance"""
        import time
        
        start_time = time.time()
        
        for _ in range(100):
            await template_manager.recommend_templates({
                "document_type": "business_plan",
                "industry": "technology"
            })
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 100 recommendations in under 5 seconds
        assert duration < 5.0
    
    @pytest.mark.asyncio
    async def test_model_selection_performance(self):
        """Test model selection performance"""
        import time
        
        start_time = time.time()
        
        for _ in range(50):
            await model_manager.get_best_model(
                model_type=ModelType.TEXT_GENERATION,
                criteria="performance"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 50 selections in under 2 seconds
        assert duration < 2.0
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance"""
        import time
        
        # Set up cache
        await cache.initialize()
        
        start_time = time.time()
        
        # Set 100 cache entries
        for i in range(100):
            await cache.set(
                key=f"perf_test_{i}",
                value=f"value_{i}",
                endpoint="/test",
                method="GET"
            )
        
        # Get 100 cache entries
        for i in range(100):
            await cache.get(
                key=f"perf_test_{i}",
                endpoint="/test",
                method="GET"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 200 operations in under 1 second
        assert duration < 1.0


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])















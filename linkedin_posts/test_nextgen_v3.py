"""
🧪 Next-Generation Test Suite for Ultra-Optimized LinkedIn Posts Optimization v3.0
===============================================================================

Comprehensive testing for all revolutionary v3.0 components with performance benchmarks.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from ultra_optimized_linkedin_optimizer_v3 import (
    NextGenLinkedInService,
    ContentData,
    ContentType,
    OptimizationStrategy,
    ContentMetrics,
    Language,
    ABTestConfig,
    RealTimeLearningEngine,
    ABTestingEngine,
    MultiLanguageOptimizer,
    DistributedProcessingEngine,
    AdvancedPerformanceMonitor
)

# Test data
SAMPLE_CONTENT = "Just completed an amazing AI project! #artificialintelligence #machinelearning"

class TestRealTimeLearningEngine:
    """Test real-time learning engine functionality."""
    
    def test_learning_engine_initialization(self):
        """Test learning engine initialization."""
        engine = RealTimeLearningEngine()
        assert engine.insights_buffer.maxlen == 10000
        assert engine.learning_rate == 0.01
        assert engine.batch_size == 100
        assert engine.update_frequency == 50
    
    def test_add_performance_data(self):
        """Test adding performance data for learning."""
        engine = RealTimeLearningEngine()
        metrics = ContentMetrics(views=1000, likes=50, shares=10, comments=5)
        
        engine.add_performance_data("test_hash", metrics, OptimizationStrategy.ENGAGEMENT)
        
        assert "test_hash" in engine.performance_history
        assert len(engine.performance_history["test_hash"]) == 1
        assert engine.performance_history["test_hash"][0]["strategy"] == OptimizationStrategy.ENGAGEMENT
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        engine = RealTimeLearningEngine()
        metrics = ContentMetrics(views=1000, likes=50, shares=10, comments=5)
        
        # Add two performance records
        engine.add_performance_data("test_hash", metrics, OptimizationStrategy.ENGAGEMENT)
        engine.add_performance_data("test_hash", metrics, OptimizationStrategy.ENGAGEMENT)
        
        # Should have generated an insight
        insights = engine.get_recent_insights()
        assert len(insights) > 0
    
    def test_get_performance_trends(self):
        """Test performance trends retrieval."""
        engine = RealTimeLearningEngine()
        metrics = ContentMetrics(views=1000, likes=50, shares=10, comments=5)
        
        # Add multiple performance records
        for i in range(10):
            engine.add_performance_data(f"hash_{i}", metrics, OptimizationStrategy.ENGAGEMENT)
        
        trends = engine.get_performance_trends()
        assert len(trends) > 0

class TestABTestingEngine:
    """Test A/B testing engine functionality."""
    
    def test_ab_test_creation(self):
        """Test A/B test creation."""
        engine = ABTestingEngine()
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B",
            description="Testing A/B",
            variants=["A", "B"],
            traffic_split=[0.5, 0.5],
            duration_days=7,
            success_metrics=["engagement"]
        )
        
        test_id = engine.create_test(config)
        assert test_id == "test_001"
        assert test_id in engine.active_tests
    
    def test_traffic_allocation(self):
        """Test traffic allocation to variants."""
        engine = ABTestingEngine()
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B",
            description="Testing A/B",
            variants=["A", "B"],
            traffic_split=[0.5, 0.5],
            duration_days=7,
            success_metrics=["engagement"]
        )
        
        engine.create_test(config)
        
        # Test multiple allocations
        variants = set()
        for i in range(100):
            variant = engine.allocate_traffic("test_001", f"user_{i}")
            variants.add(variant)
        
        # Should have both variants
        assert "A" in variants
        assert "B" in variants
    
    def test_impression_recording(self):
        """Test impression recording."""
        engine = ABTestingEngine()
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B",
            description="Testing A/B",
            variants=["A", "B"],
            traffic_split=[0.5, 0.5],
            duration_days=7,
            success_metrics=["engagement"]
        )
        
        engine.create_test(config)
        engine.record_impression("test_001", "A", "content_hash")
        
        test = engine.active_tests["test_001"]
        assert test["variants"]["A"]["impressions"] == 1
    
    def test_conversion_recording(self):
        """Test conversion recording."""
        engine = ABTestingEngine()
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B",
            description="Testing A/B",
            variants=["A", "B"],
            traffic_split=[0.5, 0.5],
            duration_days=7,
            success_metrics=["engagement"]
        )
        
        engine.create_test(config)
        engine.record_impression("test_001", "A", "content_hash")
        engine.record_conversion("test_001", "A", "content_hash", 1.0)
        
        test = engine.active_tests["test_001"]
        assert test["variants"]["A"]["conversions"] == 1.0
    
    def test_test_results(self):
        """Test A/B test results retrieval."""
        engine = ABTestingEngine()
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B",
            description="Testing A/B",
            variants=["A", "B"],
            traffic_split=[0.5, 0.5],
            duration_days=7,
            success_metrics=["engagement"]
        )
        
        engine.create_test(config)
        engine.record_impression("test_001", "A", "content_hash")
        engine.record_conversion("test_001", "A", "content_hash", 1.0)
        
        results = engine.get_test_results("test_001")
        assert "A" in results
        assert results["A"]["conversion_rate"] == 1.0
    
    def test_statistical_significance(self):
        """Test statistical significance detection."""
        engine = ABTestingEngine()
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B",
            description="Testing A/B",
            variants=["A", "B"],
            traffic_split=[0.5, 0.5],
            duration_days=7,
            success_metrics=["engagement"]
        )
        
        engine.create_test(config)
        
        # Add significant difference
        for i in range(100):
            engine.record_impression("test_001", "A", f"content_{i}")
            engine.record_impression("test_001", "B", f"content_{i}")
        
        for i in range(50):
            engine.record_conversion("test_001", "A", f"content_{i}")
        
        # B has no conversions, A has 50% conversion rate
        is_significant = engine.is_test_significant("test_001")
        assert is_significant

class TestMultiLanguageOptimizer:
    """Test multi-language optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test multi-language optimizer initialization."""
        optimizer = MultiLanguageOptimizer()
        assert hasattr(optimizer, 'translation_models')
        assert hasattr(optimizer, 'language_specific_features')
    
    def test_language_optimization_spanish(self):
        """Test Spanish language optimization."""
        optimizer = MultiLanguageOptimizer()
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        optimizations = optimizer.optimize_for_language(content, Language.SPANISH)
        
        assert optimizations['language'] == 'es'
        assert len(optimizations['cultural_adaptations']) > 0
        assert len(optimizations['localized_hashtags']) > 0
        assert len(optimizations['timing_recommendations']) > 0
        assert "#profesional" in optimizations['localized_hashtags']
    
    def test_language_optimization_french(self):
        """Test French language optimization."""
        optimizer = MultiLanguageOptimizer()
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        optimizations = optimizer.optimize_for_language(content, Language.FRENCH)
        
        assert optimizations['language'] == 'fr'
        assert "#professionnel" in optimizations['localized_hashtags']
        assert "#entreprise" in optimizations['localized_hashtags']
    
    def test_language_optimization_german(self):
        """Test German language optimization."""
        optimizer = MultiLanguageOptimizer()
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        optimizations = optimizer.optimize_for_language(content, Language.GERMAN)
        
        assert optimizations['language'] == 'de'
        assert "#beruflich" in optimizations['localized_hashtags']
        assert "#unternehmen" in optimizations['localized_hashtags']
    
    def test_language_optimization_chinese(self):
        """Test Chinese language optimization."""
        optimizer = MultiLanguageOptimizer()
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        optimizations = optimizer.optimize_for_language(content, Language.CHINESE)
        
        assert optimizations['language'] == 'zh'
        assert "#专业" in optimizations['localized_hashtags']
        assert "#商业" in optimizations['localized_hashtags']
    
    def test_language_optimization_japanese(self):
        """Test Japanese language optimization."""
        optimizer = MultiLanguageOptimizer()
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        optimizations = optimizer.optimize_for_language(content, Language.JAPANESE)
        
        assert optimizations['language'] == 'ja'
        assert "#プロフェッショナル" in optimizations['localized_hashtags']
        assert "#ビジネス" in optimizations['localized_hashtags']
    
    @pytest.mark.asyncio
    async def test_content_translation(self):
        """Test content translation functionality."""
        optimizer = MultiLanguageOptimizer()
        
        # Test same language (no translation needed)
        translated = await optimizer.translate_content(
            "Hello world", Language.ENGLISH, Language.ENGLISH
        )
        assert translated == "Hello world"
        
        # Note: Actual translation testing would require model loading
        # This is a placeholder for when models are available

class TestDistributedProcessingEngine:
    """Test distributed processing engine functionality."""
    
    @pytest.mark.asyncio
    async def test_worker_management(self):
        """Test worker start and stop functionality."""
        engine = DistributedProcessingEngine()
        
        # Start workers
        await engine.start_workers()
        assert engine.is_running
        assert len(engine.workers) == 4
        
        # Stop workers
        await engine.stop_workers()
        assert not engine.is_running
    
    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission and retrieval."""
        engine = DistributedProcessingEngine()
        await engine.start_workers()
        
        # Submit task
        task = {'task_id': 'test_1', 'worker_id': 'worker-1'}
        await engine.submit_task(task)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Get results
        results = await engine.get_results()
        assert len(results) > 0
        assert results[0]['task_id'] == 'test_1'
        
        await engine.stop_workers()

class TestAdvancedPerformanceMonitor:
    """Test advanced performance monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = AdvancedPerformanceMonitor()
        assert monitor.metrics == {}
        assert monitor.start_time > 0
        assert monitor.gpu_metrics == {}
        assert monitor.system_metrics == {}
        assert monitor.real_time_metrics.maxlen == 1000
    
    def test_operation_timing(self):
        """Test operation timing functionality."""
        monitor = AdvancedPerformanceMonitor()
        
        monitor.start_operation("test_op")
        time.sleep(0.1)  # Simulate work
        metrics = monitor.end_operation("test_op")
        
        assert "test_op" in monitor.metrics
        assert metrics["duration"] > 0
        assert monitor.metrics["test_op"]["duration"] > 0
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        monitor = AdvancedPerformanceMonitor()
        monitor.start_operation("test_op")
        time.sleep(0.1)
        monitor.end_operation("test_op")
        
        stats = monitor.get_stats()
        assert "total_uptime" in stats
        assert "operations" in stats
        assert "averages" in stats
        assert "system" in stats
        assert "real_time" in stats
        assert "test_op" in stats["operations"]
    
    def test_performance_alerts(self):
        """Test performance alert generation."""
        monitor = AdvancedPerformanceMonitor()
        
        # Simulate slow operation
        monitor.start_operation("slow_op")
        time.sleep(0.1)
        monitor.end_operation("slow_op")
        
        # Modify duration to trigger alert
        monitor.metrics["slow_op"]["duration"] = 10.0  # Very slow
        
        alerts = monitor.get_performance_alerts()
        assert len(alerts) > 0
        assert any(alert["type"] == "slow_response" for alert in alerts)

class TestContentDataEnhanced:
    """Test enhanced content data structures."""
    
    def test_content_data_creation_enhanced(self):
        """Test enhanced ContentData creation."""
        content = ContentData(
            id="test_1",
            content="Test content",
            content_type=ContentType.POST,
            target_audience=["professionals"],
            industry="technology",
            language=Language.SPANISH,
            campaign_id="campaign_001",
            ab_test_id="test_001"
        )
        
        assert content.id == "test_1"
        assert content.content == "Test content"
        assert content.content_type == ContentType.POST
        assert content.target_audience == ["professionals"]
        assert content.industry == "technology"
        assert content.language == Language.SPANISH
        assert content.campaign_id == "campaign_001"
        assert content.ab_test_id == "test_001"
        assert content.metrics is not None
        assert content.version == "v3.0"
    
    def test_content_hash_generation_enhanced(self):
        """Test enhanced content hash generation."""
        content1 = ContentData(
            id="test_1",
            content="Same content",
            content_type=ContentType.POST,
            hashtags=["#test"],
            industry="technology",
            language=Language.ENGLISH
        )
        
        content2 = ContentData(
            id="test_2",
            content="Same content",
            content_type=ContentType.POST,
            hashtags=["#test"],
            industry="technology",
            language=Language.ENGLISH
        )
        
        assert content1.get_content_hash() == content2.get_content_hash()
    
    def test_metrics_calculation_enhanced(self):
        """Test enhanced metrics calculation."""
        metrics = ContentMetrics(
            views=1000,
            likes=50,
            shares=10,
            comments=5,
            clicks=15,
            saves=3
        )
        
        engagement_rate = metrics.calculate_engagement_rate()
        viral_coefficient = metrics.calculate_viral_coefficient()
        reach_score = metrics.calculate_reach_score()
        conversion_rate = metrics.calculate_conversion_rate()
        retention_score = metrics.calculate_retention_score()
        influence_score = metrics.calculate_influence_score()
        
        assert engagement_rate == 8.3  # (50+10+5+15+3)/1000 * 100
        assert viral_coefficient == 0.025  # (10*2+5)/1000
        assert reach_score == 0.085  # (10*3+5*2+50)/1000
        assert conversion_rate == 1.5  # 15/1000 * 100
        assert retention_score == 0.013  # (3+10)/1000
        assert influence_score == 0.135  # (10*4+5*3+50*2)/1000

class TestNextGenLinkedInService:
    """Test the next-generation service class."""
    
    @pytest.fixture
    def service(self):
        return NextGenLinkedInService()
    
    @pytest.mark.asyncio
    async def test_service_initialization_enhanced(self, service):
        """Test enhanced service initialization."""
        assert service.monitor is not None
        assert service.cache == {}
        assert service.error_log == []
        assert service.cache_size == 5000
        assert service.max_workers == 8
        assert service.enable_gpu in [True, False]
        assert service.enable_mixed_precision in [True, False]
        assert service.enable_distributed in [True, False]
        
        # Check v3.0 components
        assert hasattr(service, 'learning_engine')
        assert hasattr(service, 'ab_testing_engine')
        assert hasattr(service, 'multi_language_optimizer')
        assert hasattr(service, 'distributed_engine')
    
    @pytest.mark.asyncio
    async def test_string_content_optimization_enhanced(self, service):
        """Test enhanced optimization with string content."""
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert result.original_content.content == SAMPLE_CONTENT
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0
        assert result.processing_time > 0
        assert result.performance_metrics is not None
        assert result.model_used == "nextgen_transformer_v3"
    
    @pytest.mark.asyncio
    async def test_enhanced_strategies(self, service):
        """Test enhanced optimization strategies."""
        strategies = [
            OptimizationStrategy.ENGAGEMENT,
            OptimizationStrategy.REACH,
            OptimizationStrategy.CLICKS,
            OptimizationStrategy.SHARES,
            OptimizationStrategy.COMMENTS,
            OptimizationStrategy.BRAND_AWARENESS,
            OptimizationStrategy.LEAD_GENERATION,
            OptimizationStrategy.CONVERSION,
            OptimizationStrategy.RETENTION,
            OptimizationStrategy.INFLUENCE
        ]
        
        for strategy in strategies:
            result = await service.optimize_linkedin_post(
                SAMPLE_CONTENT, 
                strategy
            )
            
            assert result.optimization_score >= 0
            assert result.confidence_score >= 0
            assert len(result.improvements) > 0
            assert result.performance_metrics is not None
    
    @pytest.mark.asyncio
    async def test_multi_language_optimization(self, service):
        """Test multi-language optimization."""
        content = ContentData(
            id="test_1",
            content="AI breakthrough in technology",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        result = await service.optimize_linkedin_post(
            content,
            OptimizationStrategy.ENGAGEMENT,
            target_language=Language.SPANISH
        )
        
        assert result.language_optimizations is not None
        assert result.optimized_content.language == Language.SPANISH
        assert "Spanish" in str(result.language_optimizations)
    
    @pytest.mark.asyncio
    async def test_ab_testing_integration(self, service):
        """Test A/B testing integration."""
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT,
            OptimizationStrategy.ENGAGEMENT,
            enable_ab_testing=True
        )
        
        assert result.ab_test_results is not None
        assert "test_id" in result.ab_test_results
        assert "variant" in result.ab_test_results
        assert "config" in result.ab_test_results
    
    @pytest.mark.asyncio
    async def test_real_time_learning_integration(self, service):
        """Test real-time learning integration."""
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT,
            OptimizationStrategy.ENGAGEMENT,
            enable_learning=True
        )
        
        # Check if learning insights are available
        insights = await service.get_learning_insights()
        assert isinstance(insights, list)
    
    @pytest.mark.asyncio
    async def test_enhanced_caching(self, service):
        """Test enhanced caching functionality."""
        # First optimization
        result1 = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        # Second optimization (should be cached)
        result2 = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        # Results should be the same
        assert result1.optimization_score == result2.optimization_score
        assert result2.processing_time < 0.01  # Cache hit should be very fast
    
    @pytest.mark.asyncio
    async def test_service_methods(self, service):
        """Test service utility methods."""
        # Test learning insights
        insights = await service.get_learning_insights()
        assert isinstance(insights, list)
        
        # Test performance trends
        trends = await service.get_performance_trends()
        assert isinstance(trends, dict)
        
        # Test performance alerts
        alerts = await service.get_performance_alerts()
        assert isinstance(alerts, list)
    
    @pytest.mark.asyncio
    async def test_service_shutdown(self, service):
        """Test service graceful shutdown."""
        await service.shutdown()
        # Service should be shutdown gracefully

class TestPerformanceBenchmarksNextGen:
    """Next-generation performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_single_optimization_performance_enhanced(self):
        """Benchmark enhanced single optimization performance."""
        service = NextGenLinkedInService()
        
        start_time = time.time()
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 5.0  # 5 seconds max for next-generation system
        assert result.performance_metrics is not None
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_optimization_performance_enhanced(self):
        """Benchmark enhanced batch optimization performance."""
        service = NextGenLinkedInService()
        
        contents = [f"Test content {i} #test{i}" for i in range(10)]
        
        start_time = time.time()
        results = []
        for content in contents:
            result = await service.optimize_linkedin_post(
                content, 
                OptimizationStrategy.ENGAGEMENT
            )
            results.append(result)
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 20.0  # 20 seconds max for enhanced batch
        assert len(results) == 10
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_language_performance(self):
        """Benchmark multi-language optimization performance."""
        service = NextGenLinkedInService()
        
        languages = [Language.SPANISH, Language.FRENCH, Language.GERMAN]
        content = ContentData(
            id="test",
            content="AI breakthrough in technology",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        start_time = time.time()
        results = []
        for language in languages:
            result = await service.optimize_linkedin_post(
                content,
                OptimizationStrategy.ENGAGEMENT,
                target_language=language
            )
            results.append(result)
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 15.0  # 15 seconds max for multi-language
        assert len(results) == 3
        
        await service.shutdown()

class TestErrorHandlingNextGen:
    """Test next-generation error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_content_enhanced(self):
        """Test handling of empty content."""
        service = NextGenLinkedInService()
        
        with pytest.raises(Exception):
            await service.optimize_linkedin_post("", OptimizationStrategy.ENGAGEMENT)
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_very_long_content_enhanced(self):
        """Test handling of very long content."""
        service = NextGenLinkedInService()
        
        long_content = "Test content " * 2000  # Very long content
        
        result = await service.optimize_linkedin_post(
            long_content, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_special_characters_enhanced(self):
        """Test handling of special characters."""
        service = NextGenLinkedInService()
        
        special_content = "Test content with émojis 🚀 and special chars: @#$%^&*()"
        
        result = await service.optimize_linkedin_post(
            special_content, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0
        
        await service.shutdown()

class TestIntegrationNextGen:
    """Next-generation integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_enhanced(self):
        """Test complete next-generation optimization workflow."""
        service = NextGenLinkedInService()
        
        # Test content with various elements
        test_content = ContentData(
            id="integration_test_nextgen",
            content="Amazing AI breakthrough! This will revolutionize the industry.",
            content_type=ContentType.POST,
            hashtags=["#ai", "#innovation"],
            mentions=["@tech_leader"],
            links=["https://research-paper.com"],
            media_urls=["https://image.jpg"],
            industry="technology",
            target_audience=["professionals", "entrepreneurs"],
            language=Language.ENGLISH
        )
        
        # Test enhanced strategies
        strategies = [
            OptimizationStrategy.ENGAGEMENT,
            OptimizationStrategy.REACH,
            OptimizationStrategy.CLICKS,
            OptimizationStrategy.SHARES,
            OptimizationStrategy.COMMENTS,
            OptimizationStrategy.BRAND_AWARENESS,
            OptimizationStrategy.LEAD_GENERATION,
            OptimizationStrategy.CONVERSION,
            OptimizationStrategy.RETENTION,
            OptimizationStrategy.INFLUENCE
        ]
        
        results = []
        for strategy in strategies:
            result = await service.optimize_linkedin_post(
                test_content, 
                strategy,
                enable_ab_testing=True,
                enable_learning=True
            )
            
            # Verify enhanced result structure
            assert result.original_content.id == test_content.id
            assert result.optimization_score >= 0
            assert result.confidence_score >= 0
            assert result.processing_time > 0
            assert len(result.improvements) > 0
            assert result.model_used == "nextgen_transformer_v3"
            assert result.performance_metrics is not None
            assert result.ab_test_results is not None
            assert result.learning_insights is not None
        
        # Verify different strategies produce different results
        optimization_scores = [r.optimization_score for r in results]
        assert len(set(optimization_scores)) > 1  # Should have some variation
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_language_workflow(self):
        """Test complete multi-language workflow."""
        service = NextGenLinkedInService()
        
        content = ContentData(
            id="multilang_test",
            content="AI breakthrough in machine learning technology",
            content_type=ContentType.POST,
            language=Language.ENGLISH
        )
        
        languages = [Language.SPANISH, Language.FRENCH, Language.GERMAN, Language.CHINESE]
        
        for language in languages:
            result = await service.optimize_linkedin_post(
                content,
                OptimizationStrategy.ENGAGEMENT,
                target_language=language
            )
            
            assert result.language_optimizations is not None
            assert result.optimized_content.language == language
            assert result.optimization_score >= 0
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self):
        """Test complete A/B testing workflow."""
        service = NextGenLinkedInService()
        
        content = ContentData(
            id="abtest_test",
            content="Testing A/B optimization workflow",
            content_type=ContentType.POST
        )
        
        # Test with A/B testing enabled
        result = await service.optimize_linkedin_post(
            content,
            OptimizationStrategy.ENGAGEMENT,
            enable_ab_testing=True
        )
        
        assert result.ab_test_results is not None
        test_id = result.ab_test_results['test_id']
        
        # Get test results
        test_results = await service.get_ab_test_results(test_id)
        assert isinstance(test_results, dict)
        
        await service.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

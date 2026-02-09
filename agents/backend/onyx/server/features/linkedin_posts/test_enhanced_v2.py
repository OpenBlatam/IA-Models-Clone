"""
🧪 Enhanced Test Suite for Ultra-Optimized LinkedIn Posts Optimization v2.0
==========================================================================

Comprehensive testing for all enhanced components with performance benchmarks.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from ultra_optimized_linkedin_optimizer_v2 import (
    EnhancedLinkedInService,
    ContentData,
    ContentType,
    OptimizationStrategy,
    ContentMetrics,
    AdvancedTransformerAnalyzer,
    EnhancedPerformanceMonitor
)

# Test data
SAMPLE_CONTENT = "Just completed an amazing AI project! #artificialintelligence #machinelearning"

class TestEnhancedPerformanceMonitor:
    """Test enhanced performance monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test enhanced monitor initialization."""
        monitor = EnhancedPerformanceMonitor()
        assert monitor.metrics == {}
        assert monitor.start_time > 0
        assert monitor.gpu_metrics == {}
        assert monitor.system_metrics == {}
    
    def test_operation_timing_enhanced(self):
        """Test enhanced operation timing functionality."""
        monitor = EnhancedPerformanceMonitor()
        
        monitor.start_operation("test_op")
        time.sleep(0.1)  # Simulate work
        metrics = monitor.end_operation("test_op")
        
        assert "test_op" in monitor.metrics
        assert metrics["duration"] > 0
        assert monitor.metrics["test_op"]["duration"] > 0
    
    def test_get_stats_enhanced(self):
        """Test enhanced statistics retrieval."""
        monitor = EnhancedPerformanceMonitor()
        monitor.start_operation("test_op")
        time.sleep(0.1)
        monitor.end_operation("test_op")
        
        stats = monitor.get_stats()
        assert "total_uptime" in stats
        assert "operations" in stats
        assert "averages" in stats
        assert "system" in stats
        assert "test_op" in stats["operations"]

class TestContentDataEnhanced:
    """Test enhanced content data structures."""
    
    def test_content_data_creation_enhanced(self):
        """Test enhanced ContentData creation."""
        content = ContentData(
            id="test_1",
            content="Test content",
            content_type=ContentType.POST,
            target_audience=["professionals"],
            industry="technology"
        )
        
        assert content.id == "test_1"
        assert content.content == "Test content"
        assert content.content_type == ContentType.POST
        assert content.target_audience == ["professionals"]
        assert content.industry == "technology"
        assert content.metrics is not None
    
    def test_content_hash_generation_enhanced(self):
        """Test enhanced content hash generation."""
        content1 = ContentData(
            id="test_1",
            content="Same content",
            content_type=ContentType.POST,
            hashtags=["#test"],
            industry="technology"
        )
        
        content2 = ContentData(
            id="test_2",
            content="Same content",
            content_type=ContentType.POST,
            hashtags=["#test"],
            industry="technology"
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
        
        assert engagement_rate == 8.3  # (50+10+5+15+3)/1000 * 100
        assert viral_coefficient == 0.025  # (10*2+5)/1000
        assert reach_score == 0.085  # (10*3+5*2+50)/1000

class TestAdvancedTransformerAnalyzer:
    """Test advanced transformer analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        return AdvancedTransformerAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert hasattr(analyzer, 'device')
        assert hasattr(analyzer, 'models')
        assert hasattr(analyzer, 'tokenizers')
    
    @pytest.mark.asyncio
    async def test_content_analysis_enhanced(self, analyzer):
        """Test enhanced content analysis."""
        content = ContentData(
            id="test_1",
            content="Amazing AI breakthrough in machine learning! #ai #ml",
            content_type=ContentType.POST,
            hashtags=["#ai", "#ml"],
            industry="technology",
            target_audience=["professionals"]
        )
        
        analysis = await analyzer.analyze_content(content)
        
        assert 'sentiment' in analysis
        assert 'sentiment_score' in analysis
        assert 'classification' in analysis
        assert 'classification_score' in analysis
        assert 'readability_score' in analysis
        assert 'complexity_score' in analysis
        assert 'industry_relevance' in analysis
        assert 'audience_targeting' in analysis
    
    @pytest.mark.asyncio
    async def test_feature_extraction_enhanced(self, analyzer):
        """Test enhanced feature extraction."""
        content = ContentData(
            id="test_1",
            content="Test content with hashtags #test #example",
            content_type=ContentType.POST,
            hashtags=["#test", "#example"]
        )
        
        features = await analyzer.extract_features(content)
        
        assert len(features) == 150  # Enhanced feature count
        assert all(isinstance(f, float) for f in features)
        assert all(0 <= f <= 1 for f in features)

class TestEnhancedLinkedInService:
    """Test the enhanced service class."""
    
    @pytest.fixture
    def service(self):
        return EnhancedLinkedInService()
    
    @pytest.mark.asyncio
    async def test_service_initialization_enhanced(self, service):
        """Test enhanced service initialization."""
        assert service.monitor is not None
        assert service.cache == {}
        assert service.error_log == []
        assert service.cache_size == 2000
        assert service.max_workers == 4
    
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
            OptimizationStrategy.LEAD_GENERATION
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

class TestPerformanceBenchmarksEnhanced:
    """Enhanced performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_single_optimization_performance_enhanced(self):
        """Benchmark enhanced single optimization performance."""
        service = EnhancedLinkedInService()
        
        start_time = time.time()
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 10.0  # 10 seconds max for enhanced system
        assert result.performance_metrics is not None
    
    @pytest.mark.asyncio
    async def test_batch_optimization_performance_enhanced(self):
        """Benchmark enhanced batch optimization performance."""
        service = EnhancedLinkedInService()
        
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
        assert total_time < 30.0  # 30 seconds max for enhanced batch
        assert len(results) == 10

class TestErrorHandlingEnhanced:
    """Test enhanced error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_content_enhanced(self):
        """Test handling of empty content."""
        service = EnhancedLinkedInService()
        
        with pytest.raises(Exception):
            await service.optimize_linkedin_post("", OptimizationStrategy.ENGAGEMENT)
    
    @pytest.mark.asyncio
    async def test_very_long_content_enhanced(self):
        """Test handling of very long content."""
        service = EnhancedLinkedInService()
        
        long_content = "Test content " * 2000  # Very long content
        
        result = await service.optimize_linkedin_post(
            long_content, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0
    
    @pytest.mark.asyncio
    async def test_special_characters_enhanced(self):
        """Test handling of special characters."""
        service = EnhancedLinkedInService()
        
        special_content = "Test content with émojis 🚀 and special chars: @#$%^&*()"
        
        result = await service.optimize_linkedin_post(
            special_content, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0

class TestIntegrationEnhanced:
    """Enhanced integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_enhanced(self):
        """Test complete enhanced optimization workflow."""
        service = EnhancedLinkedInService()
        
        # Test content with various elements
        test_content = ContentData(
            id="integration_test_enhanced",
            content="Amazing AI breakthrough! This will revolutionize the industry.",
            content_type=ContentType.POST,
            hashtags=["#ai", "#innovation"],
            mentions=["@tech_leader"],
            links=["https://research-paper.com"],
            media_urls=["https://image.jpg"],
            industry="technology",
            target_audience=["professionals", "entrepreneurs"]
        )
        
        # Test enhanced strategies
        strategies = [
            OptimizationStrategy.ENGAGEMENT,
            OptimizationStrategy.REACH,
            OptimizationStrategy.CLICKS,
            OptimizationStrategy.SHARES,
            OptimizationStrategy.COMMENTS,
            OptimizationStrategy.BRAND_AWARENESS,
            OptimizationStrategy.LEAD_GENERATION
        ]
        
        results = []
        for strategy in strategies:
            result = await service.optimize_linkedin_post(test_content, strategy)
            results.append(result)
            
            # Verify enhanced result structure
            assert result.original_content.id == test_content.id
            assert result.optimization_score >= 0
            assert result.confidence_score >= 0
            assert result.processing_time > 0
            assert len(result.improvements) > 0
            assert result.model_used in ["advanced_transformer", "fallback"]
            assert result.performance_metrics is not None
        
        # Verify different strategies produce different results
        optimization_scores = [r.optimization_score for r in results]
        assert len(set(optimization_scores)) > 1  # Should have some variation

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

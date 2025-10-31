"""
🧪 Test Suite for Ultra-Optimized LinkedIn Posts Optimization
===========================================================

Comprehensive testing for all components with performance benchmarks.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from ultra_optimized_linkedin_optimizer import (
    UltraOptimizedLinkedInService,
    ContentData,
    ContentType,
    OptimizationStrategy,
    ContentMetrics
)

# Test data
SAMPLE_CONTENT = "Just completed an amazing AI project! #artificialintelligence #machinelearning"

class TestUltraOptimizedLinkedInService:
    """Test the main service class."""
    
    @pytest.fixture
    def service(self):
        return UltraOptimizedLinkedInService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initialization."""
        assert service.monitor is not None
        assert service.cache == {}
        assert service.error_log == []
    
    @pytest.mark.asyncio
    async def test_string_content_optimization(self, service):
        """Test optimization with string content."""
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert result.original_content.content == SAMPLE_CONTENT
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_different_strategies(self, service):
        """Test different optimization strategies."""
        strategies = [
            OptimizationStrategy.ENGAGEMENT,
            OptimizationStrategy.REACH,
            OptimizationStrategy.CLICKS,
            OptimizationStrategy.SHARES,
            OptimizationStrategy.COMMENTS
        ]
        
        for strategy in strategies:
            result = await service.optimize_linkedin_post(
                SAMPLE_CONTENT, 
                strategy
            )
            
            assert result.optimization_score >= 0
            assert result.confidence_score >= 0
            assert len(result.improvements) > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check functionality."""
        health = await service.health_check()
        
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        assert health["status"] in ["healthy", "degraded"]

class TestPerformanceBenchmarks:
    """Performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_single_optimization_performance(self):
        """Benchmark single optimization performance."""
        service = UltraOptimizedLinkedInService()
        
        start_time = time.time()
        result = await service.optimize_linkedin_post(
            SAMPLE_CONTENT, 
            OptimizationStrategy.ENGAGEMENT
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 5.0  # 5 seconds max
    
    @pytest.mark.asyncio
    async def test_batch_optimization_performance(self):
        """Benchmark batch optimization performance."""
        service = UltraOptimizedLinkedInService()
        
        contents = [f"Test content {i} #test{i}" for i in range(5)]
        
        start_time = time.time()
        results = await service.batch_optimize(
            contents, 
            OptimizationStrategy.ENGAGEMENT
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 10.0  # 10 seconds max
        assert len(results) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

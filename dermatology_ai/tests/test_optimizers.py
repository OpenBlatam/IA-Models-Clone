"""
Tests for Optimizers
Tests for PerformanceOptimizer and MLOptimizer
"""

import pytest
from unittest.mock import Mock, AsyncMock
import numpy as np

from core.performance_optimizer import PerformanceOptimizer
from core.ml_optimizer import MLOptimizer


class TestPerformanceOptimizer:
    """Tests for PerformanceOptimizer"""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create performance optimizer"""
        return PerformanceOptimizer()
    
    def test_optimize_query(self, performance_optimizer):
        """Test query optimization"""
        query = {
            "user_id": "user-123",
            "limit": 100,
            "offset": 0
        }
        
        optimized = performance_optimizer.optimize_query(query)
        
        assert optimized is not None
        assert isinstance(optimized, dict)
    
    def test_optimize_cache_strategy(self, performance_optimizer):
        """Test cache strategy optimization"""
        cache_config = {
            "ttl": 3600,
            "max_size": 1000
        }
        
        optimized = performance_optimizer.optimize_cache(cache_config)
        
        assert optimized is not None
        assert isinstance(optimized, dict)
    
    def test_optimize_database_connection(self, performance_optimizer):
        """Test database connection optimization"""
        pool_config = {
            "min_size": 5,
            "max_size": 20
        }
        
        optimized = performance_optimizer.optimize_connection_pool(pool_config)
        
        assert optimized is not None
        assert isinstance(optimized, dict)
    
    def test_get_optimization_recommendations(self, performance_optimizer):
        """Test getting optimization recommendations"""
        metrics = {
            "avg_response_time": 0.5,
            "cache_hit_rate": 0.7,
            "db_query_time": 0.2
        }
        
        recommendations = performance_optimizer.get_recommendations(metrics)
        
        assert isinstance(recommendations, list)
        # Should provide optimization suggestions
    
    def test_optimize_batch_size(self, performance_optimizer):
        """Test batch size optimization"""
        current_batch_size = 10
        processing_time = 0.5
        
        optimal_size = performance_optimizer.optimize_batch_size(
            current_batch_size,
            processing_time
        )
        
        assert optimal_size > 0
        assert isinstance(optimal_size, int)


class TestMLOptimizer:
    """Tests for MLOptimizer"""
    
    @pytest.fixture
    def ml_optimizer(self):
        """Create ML optimizer"""
        return MLOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimize_model(self, ml_optimizer):
        """Test model optimization"""
        mock_model = Mock()
        
        optimized_model = await ml_optimizer.optimize(mock_model)
        
        assert optimized_model is not None
    
    @pytest.mark.asyncio
    async def test_quantize_model(self, ml_optimizer):
        """Test model quantization"""
        mock_model = Mock()
        
        quantized_model = await ml_optimizer.quantize(mock_model)
        
        assert quantized_model is not None
    
    @pytest.mark.asyncio
    async def test_prune_model(self, ml_optimizer):
        """Test model pruning"""
        mock_model = Mock()
        
        pruned_model = await ml_optimizer.prune(mock_model, sparsity=0.5)
        
        assert pruned_model is not None
    
    def test_optimize_hyperparameters(self, ml_optimizer):
        """Test hyperparameter optimization"""
        current_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
        
        optimized_params = ml_optimizer.optimize_hyperparameters(current_params)
        
        assert optimized_params is not None
        assert isinstance(optimized_params, dict)
    
    def test_get_model_size(self, ml_optimizer):
        """Test getting model size"""
        mock_model = Mock()
        
        size = ml_optimizer.get_model_size(mock_model)
        
        assert size is not None
        assert isinstance(size, (int, float))
        assert size >= 0
    
    @pytest.mark.asyncio
    async def test_optimize_inference_speed(self, ml_optimizer):
        """Test optimizing inference speed"""
        mock_model = Mock()
        
        optimized = await ml_optimizer.optimize_for_inference(mock_model)
        
        assert optimized is not None


class TestOptimizerIntegration:
    """Integration tests for optimizers"""
    
    @pytest.mark.asyncio
    async def test_combined_optimization(self):
        """Test combining multiple optimizations"""
        perf_optimizer = PerformanceOptimizer()
        ml_optimizer = MLOptimizer()
        
        # Optimize performance
        query = {"limit": 100}
        optimized_query = perf_optimizer.optimize_query(query)
        
        # Optimize ML model
        mock_model = Mock()
        optimized_model = await ml_optimizer.optimize(mock_model)
        
        # Both should work together
        assert optimized_query is not None
        assert optimized_model is not None




"""
Unit tests for the ads optimization layer.

This module consolidates tests for:
- Base optimizer and optimization strategies
- Optimization factory and factory pattern
- Performance, profiling, and GPU optimizers
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agents.backend.onyx.server.features.ads.optimization.base_optimizer import (
    BaseOptimizer, OptimizationStrategy, OptimizationLevel, OptimizationResult, OptimizationContext
)
from agents.backend.onyx.server.features.ads.optimization.factory import (
    OptimizationFactory, OptimizerType
)
from agents.backend.onyx.server.features.ads.optimization.performance_optimizer import (
    PerformanceOptimizer
)
from agents.backend.onyx.server.features.ads.optimization.profiling_optimizer import (
    ProfilingOptimizer
)
from agents.backend.onyx.server.features.ads.optimization.gpu_optimizer import (
    GPUOptimizer
)


class TestOptimizationStrategy:
    """Test OptimizationStrategy enum."""
    
    def test_optimization_strategy_values(self):
        """Test that all expected optimization strategy values exist."""
        assert OptimizationStrategy.CPU_OPTIMIZATION == "cpu_optimization"
        assert OptimizationStrategy.MEMORY_OPTIMIZATION == "memory_optimization"
        assert OptimizationStrategy.RESPONSE_TIME_OPTIMIZATION == "response_time_optimization"
        assert OptimizationStrategy.THROUGHPUT_OPTIMIZATION == "throughput_optimization"
        assert OptimizationStrategy.GPU_OPTIMIZATION == "gpu_optimization"
        assert OptimizationStrategy.NETWORK_OPTIMIZATION == "network_optimization"


class TestOptimizationLevel:
    """Test OptimizationLevel enum."""
    
    def test_optimization_level_values(self):
        """Test that all expected optimization level values exist."""
        assert OptimizationLevel.LIGHT == "light"
        assert OptimizationLevel.STANDARD == "standard"
        assert OptimizationLevel.AGGRESSIVE == "aggressive"
        assert OptimizationLevel.EXTREME == "extreme"


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation with valid values."""
        result = OptimizationResult(
            success=True,
            optimizer_name="test_optimizer",
            optimization_level=OptimizationLevel.STANDARD,
            improvements={
                "cpu_usage": -0.15,
                "memory_usage": -0.20,
                "response_time": -0.10
            },
            execution_time=1.5,
            timestamp=datetime.now()
        )
        assert result.success is True
        assert result.optimizer_name == "test_optimizer"
        assert result.optimization_level == OptimizationLevel.STANDARD
        assert result.improvements["cpu_usage"] == -0.15
        assert result.execution_time == 1.5
        assert result.timestamp is not None


class TestOptimizationContext:
    """Test OptimizationContext dataclass."""
    
    def test_optimization_context_creation(self):
        """Test OptimizationContext creation with valid values."""
        context = OptimizationContext(
            current_metrics={
                "cpu_usage": 0.75,
                "memory_usage": 0.80,
                "response_time": 150
            },
            target_metrics={
                "cpu_usage": 0.50,
                "memory_usage": 0.60,
                "response_time": 100
            },
            constraints={
                "max_memory": 8192,
                "max_cpu": 0.90
            },
            optimization_level=OptimizationLevel.STANDARD
        )
        assert context.current_metrics["cpu_usage"] == 0.75
        assert context.target_metrics["cpu_usage"] == 0.50
        assert context.constraints["max_memory"] == 8192
        assert context.optimization_level == OptimizationLevel.STANDARD


class TestBaseOptimizer:
    """Test BaseOptimizer abstract class."""
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer that inherits from BaseOptimizer."""
        class MockOptimizer(BaseOptimizer):
            def __init__(self):
                super().__init__("MockOptimizer")
            
            async def optimize(self, context: OptimizationContext) -> OptimizationResult:
                return OptimizationResult(
                    success=True,
                    optimizer_name=self.name,
                    optimization_level=context.optimization_level,
                    improvements={"test": 0.1},
                    execution_time=0.5,
                    timestamp=datetime.now()
                )
            
            def can_optimize(self, context: OptimizationContext) -> bool:
                return True
            
            def get_optimization_capabilities(self) -> Dict[str, Any]:
                return {
                    "strategies": [OptimizationStrategy.CPU_OPTIMIZATION],
                    "levels": [OptimizationLevel.LIGHT, OptimizationLevel.STANDARD]
                }
        
        return MockOptimizer()
    
    def test_base_optimizer_creation(self, mock_optimizer):
        """Test BaseOptimizer creation."""
        assert mock_optimizer.name == "MockOptimizer"
        assert mock_optimizer.metrics is not None
        assert mock_optimizer.statistics is not None
    
    def test_base_optimizer_metrics(self, mock_optimizer):
        """Test BaseOptimizer metrics collection."""
        # Simulate some optimization runs
        mock_optimizer.metrics["total_optimizations"] = 5
        mock_optimizer.metrics["successful_optimizations"] = 4
        mock_optimizer.metrics["failed_optimizations"] = 1
        
        assert mock_optimizer.metrics["total_optimizations"] == 5
        assert mock_optimizer.metrics["successful_optimizations"] == 4
        assert mock_optimizer.metrics["failed_optimizations"] == 1
    
    def test_base_optimizer_statistics(self, mock_optimizer):
        """Test BaseOptimizer statistics calculation."""
        # Simulate some optimization runs
        mock_optimizer.metrics["total_optimizations"] = 10
        mock_optimizer.metrics["successful_optimizations"] = 8
        mock_optimizer.metrics["total_execution_time"] = 5.0
        
        stats = mock_optimizer.get_statistics()
        
        assert stats["total_optimizations"] == 10
        assert stats["success_rate"] == 0.8
        assert stats["average_execution_time"] == 0.5
    
    @pytest.mark.asyncio
    async def test_base_optimizer_execute_optimization(self, mock_optimizer):
        """Test BaseOptimizer execute_optimization method."""
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75},
            target_metrics={"cpu_usage": 0.50},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        result = await mock_optimizer.execute_optimization(context)
        
        assert result.success is True
        assert result.optimizer_name == "MockOptimizer"
        assert result.optimization_level == OptimizationLevel.STANDARD
        
        # Verify metrics were updated
        assert mock_optimizer.metrics["total_optimizations"] == 1
        assert mock_optimizer.metrics["successful_optimizations"] == 1
    
    @pytest.mark.asyncio
    async def test_base_optimizer_execute_optimization_with_error(self, mock_optimizer):
        """Test BaseOptimizer execute_optimization method with error."""
        # Override the optimize method to raise an exception
        async def failing_optimize(context):
            raise Exception("Optimization failed")
        
        mock_optimizer.optimize = failing_optimize
        
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75},
            target_metrics={"cpu_usage": 0.50},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        result = await mock_optimizer.execute_optimization(context)
        
        assert result.success is False
        assert "Optimization failed" in result.error
        
        # Verify metrics were updated
        assert mock_optimizer.metrics["total_optimizations"] == 1
        assert mock_optimizer.metrics["failed_optimizations"] == 1
    
    def test_base_optimizer_hooks(self, mock_optimizer):
        """Test BaseOptimizer hooks system."""
        # Test pre-optimization hook
        pre_hook_called = False
        def pre_hook(context):
            nonlocal pre_hook_called
            pre_hook_called = True
        
        mock_optimizer.add_pre_optimization_hook(pre_hook)
        
        # Test post-optimization hook
        post_hook_called = False
        def post_hook(context, result):
            nonlocal post_hook_called
            post_hook_called = True
        
        mock_optimizer.add_post_optimization_hook(post_hook)
        
        # Verify hooks were added
        assert len(mock_optimizer.pre_optimization_hooks) == 1
        assert len(mock_optimizer.post_optimization_hooks) == 1


class TestOptimizerType:
    """Test OptimizerType enum."""
    
    def test_optimizer_type_values(self):
        """Test that all expected optimizer type values exist."""
        assert OptimizerType.PERFORMANCE == "performance"
        assert OptimizerType.PROFILING == "profiling"
        assert OptimizerType.GPU == "gpu"
        assert OptimizerType.MEMORY == "memory"
        assert OptimizerType.NETWORK == "network"


class TestOptimizationFactory:
    """Test OptimizationFactory class."""
    
    @pytest.fixture
    def mock_performance_optimizer(self):
        """Mock PerformanceOptimizer."""
        return Mock(spec=PerformanceOptimizer)
    
    @pytest.fixture
    def mock_profiling_optimizer(self):
        """Mock ProfilingOptimizer."""
        return Mock(spec=ProfilingOptimizer)
    
    @pytest.fixture
    def mock_gpu_optimizer(self):
        """Mock GPUOptimizer."""
        return Mock(spec=GPUOptimizer)
    
    @pytest.fixture
    def optimization_factory(self):
        """Create OptimizationFactory instance."""
        return OptimizationFactory()
    
    def test_optimization_factory_creation(self, optimization_factory):
        """Test OptimizationFactory creation."""
        assert optimization_factory.registered_optimizers == {}
        assert optimization_factory.optimizer_instances == {}
        assert optimization_factory.optimizer_configs == {}
    
    def test_optimization_factory_register_optimizer(self, optimization_factory, mock_performance_optimizer):
        """Test registering an optimizer."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        
        assert OptimizerType.PERFORMANCE in optimization_factory.registered_optimizers
        assert optimization_factory.registered_optimizers[OptimizerType.PERFORMANCE] == mock_performance_optimizer
        assert optimization_factory.optimizer_configs[OptimizerType.PERFORMANCE]["enabled"] is True
    
    def test_optimization_factory_create_optimizer(self, optimization_factory, mock_performance_optimizer):
        """Test creating an optimizer."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        
        optimizer = optimization_factory.create_optimizer(OptimizerType.PERFORMANCE)
        
        assert optimizer == mock_performance_optimizer
    
    def test_optimization_factory_create_nonexistent_optimizer(self, optimization_factory):
        """Test creating a non-existent optimizer."""
        with pytest.raises(ValueError, match="Optimizer type 'nonexistent' not found"):
            optimization_factory.create_optimizer("nonexistent")
    
    def test_optimization_factory_get_or_create_optimizer(self, optimization_factory, mock_performance_optimizer):
        """Test getting or creating an optimizer."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        
        # First call should create new instance
        optimizer1 = optimization_factory.get_or_create_optimizer(OptimizerType.PERFORMANCE)
        assert optimizer1 == mock_performance_optimizer
        
        # Second call should return existing instance
        optimizer2 = optimization_factory.get_or_create_optimizer(OptimizerType.PERFORMANCE)
        assert optimizer2 == optimizer1
    
    def test_optimization_factory_get_optimizer_info(self, optimization_factory, mock_performance_optimizer):
        """Test getting optimizer information."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True, "priority": "high"}
        )
        
        info = optimization_factory.get_optimizer_info(OptimizerType.PERFORMANCE)
        
        assert info["type"] == OptimizerType.PERFORMANCE
        assert info["enabled"] is True
        assert info["priority"] == "high"
    
    def test_optimization_factory_list_available_optimizers(self, optimization_factory, mock_performance_optimizer, mock_profiling_optimizer):
        """Test listing available optimizers."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        optimization_factory.register_optimizer(
            OptimizerType.PROFILING,
            mock_profiling_optimizer,
            {"enabled": False}
        )
        
        optimizers = optimization_factory.list_available_optimizers()
        
        assert OptimizerType.PERFORMANCE in optimizers
        assert OptimizerType.PROFILING in optimizers
        assert optimizers[OptimizerType.PERFORMANCE]["enabled"] is True
        assert optimizers[OptimizerType.PROFILING]["enabled"] is False
    
    def test_optimization_factory_can_handle_optimization(self, optimization_factory, mock_performance_optimizer):
        """Test checking if factory can handle optimization."""
        mock_performance_optimizer.can_optimize.return_value = True
        
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        
        context = OptimizationContext(
            current_metrics={},
            target_metrics={},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        can_handle = optimization_factory.can_handle_optimization(context)
        assert can_handle is True
        
        mock_performance_optimizer.can_optimize.assert_called_once_with(context)
    
    def test_optimization_factory_get_optimal_optimizer(self, optimization_factory, mock_performance_optimizer, mock_profiling_optimizer):
        """Test getting the optimal optimizer for a context."""
        mock_performance_optimizer.can_optimize.return_value = True
        mock_profiling_optimizer.can_optimize.return_value = True
        
        # Mock capabilities
        mock_performance_optimizer.get_optimization_capabilities.return_value = {
            "strategies": [OptimizationStrategy.CPU_OPTIMIZATION],
            "levels": [OptimizationLevel.STANDARD]
        }
        mock_profiling_optimizer.get_optimization_capabilities.return_value = {
            "strategies": [OptimizationStrategy.MEMORY_OPTIMIZATION],
            "levels": [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]
        }
        
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True, "priority": 1}
        )
        optimization_factory.register_optimizer(
            OptimizerType.PROFILING,
            mock_profiling_optimizer,
            {"enabled": True, "priority": 2}
        )
        
        context = OptimizationContext(
            current_metrics={},
            target_metrics={},
            constraints={},
            optimization_level=OptimizationLevel.AGGRESSIVE
        )
        
        optimal_optimizer = optimization_factory.get_optimal_optimizer(context)
        
        # Profiling optimizer should be selected as it supports aggressive level
        assert optimal_optimizer == mock_profiling_optimizer
    
    @pytest.mark.asyncio
    async def test_optimization_factory_execute_optimization(self, optimization_factory, mock_performance_optimizer):
        """Test executing optimization through factory."""
        mock_performance_optimizer.can_optimize.return_value = True
        mock_performance_optimizer.execute_optimization.return_value = OptimizationResult(
            success=True,
            optimizer_name="PerformanceOptimizer",
            optimization_level=OptimizationLevel.STANDARD,
            improvements={"cpu_usage": -0.15},
            execution_time=1.0,
            timestamp=datetime.now()
        )
        
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75},
            target_metrics={"cpu_usage": 0.50},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        result = await optimization_factory.execute_optimization(context)
        
        assert result.success is True
        assert result.optimizer_name == "PerformanceOptimizer"
        mock_performance_optimizer.execute_optimization.assert_called_once_with(context)
    
    def test_optimization_factory_cleanup(self, optimization_factory, mock_performance_optimizer):
        """Test cleanup of optimizer instances."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        
        # Create an instance
        optimization_factory.get_or_create_optimizer(OptimizerType.PERFORMANCE)
        
        # Verify instance exists
        assert OptimizerType.PERFORMANCE in optimization_factory.optimizer_instances
        
        # Cleanup
        optimization_factory.cleanup_optimizer(OptimizerType.PERFORMANCE)
        
        # Verify instance was removed
        assert OptimizerType.PERFORMANCE not in optimization_factory.optimizer_instances
    
    def test_optimization_factory_cleanup_all(self, optimization_factory, mock_performance_optimizer, mock_profiling_optimizer):
        """Test cleanup of all optimizer instances."""
        optimization_factory.register_optimizer(
            OptimizerType.PERFORMANCE,
            mock_performance_optimizer,
            {"enabled": True}
        )
        optimization_factory.register_optimizer(
            OptimizerType.PROFILING,
            mock_profiling_optimizer,
            {"enabled": True}
        )
        
        # Create instances
        optimization_factory.get_or_create_optimizer(OptimizerType.PERFORMANCE)
        optimization_factory.get_or_create_optimizer(OptimizerType.PROFILING)
        
        # Verify instances exist
        assert len(optimization_factory.optimizer_instances) == 2
        
        # Cleanup all
        optimization_factory.cleanup_all()
        
        # Verify all instances were removed
        assert len(optimization_factory.optimizer_instances) == 0


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create PerformanceOptimizer instance."""
        return PerformanceOptimizer()
    
    def test_performance_optimizer_creation(self, performance_optimizer):
        """Test PerformanceOptimizer creation."""
        assert performance_optimizer.name == "PerformanceOptimizer"
        assert performance_optimizer.can_optimize is not None
        assert performance_optimizer.get_optimization_capabilities is not None
    
    def test_performance_optimizer_can_optimize(self, performance_optimizer):
        """Test PerformanceOptimizer can_optimize method."""
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75, "memory_usage": 0.80},
            target_metrics={"cpu_usage": 0.50, "memory_usage": 0.60},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        can_optimize = performance_optimizer.can_optimize(context)
        assert can_optimize is True
    
    def test_performance_optimizer_get_capabilities(self, performance_optimizer):
        """Test PerformanceOptimizer get_optimization_capabilities method."""
        capabilities = performance_optimizer.get_optimization_capabilities()
        
        assert "strategies" in capabilities
        assert "levels" in capabilities
        assert OptimizationStrategy.CPU_OPTIMIZATION in capabilities["strategies"]
        assert OptimizationLevel.LIGHT in capabilities["levels"]
        assert OptimizationLevel.EXTREME in capabilities["levels"]
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_light_optimization(self, performance_optimizer):
        """Test PerformanceOptimizer light optimization."""
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75, "memory_usage": 0.80},
            target_metrics={"cpu_usage": 0.50, "memory_usage": 0.60},
            constraints={},
            optimization_level=OptimizationLevel.LIGHT
        )
        
        result = await performance_optimizer.optimize(context)
        
        assert result.success is True
        assert result.optimizer_name == "PerformanceOptimizer"
        assert result.optimization_level == OptimizationLevel.LIGHT
        assert "improvements" in result.improvements
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_standard_optimization(self, performance_optimizer):
        """Test PerformanceOptimizer standard optimization."""
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75, "memory_usage": 0.80},
            target_metrics={"cpu_usage": 0.50, "memory_usage": 0.60},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        result = await performance_optimizer.optimize(context)
        
        assert result.success is True
        assert result.optimization_level == OptimizationLevel.STANDARD
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_aggressive_optimization(self, performance_optimizer):
        """Test PerformanceOptimizer aggressive optimization."""
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75, "memory_usage": 0.80},
            target_metrics={"cpu_usage": 0.50, "memory_usage": 0.60},
            constraints={},
            optimization_level=OptimizationLevel.AGGRESSIVE
        )
        
        result = await performance_optimizer.optimize(context)
        
        assert result.success is True
        assert result.optimization_level == OptimizationLevel.AGGRESSIVE
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_extreme_optimization(self, performance_optimizer):
        """Test PerformanceOptimizer extreme optimization."""
        context = OptimizationContext(
            current_metrics={"cpu_usage": 0.75, "memory_usage": 0.80},
            target_metrics={"cpu_usage": 0.50, "memory_usage": 0.60},
            constraints={},
            optimization_level=OptimizationLevel.EXTREME
        )
        
        result = await performance_optimizer.optimize(context)
        
        assert result.success is True
        assert result.optimization_level == OptimizationLevel.EXTREME


class TestProfilingOptimizer:
    """Test ProfilingOptimizer class."""
    
    @pytest.fixture
    def profiling_optimizer(self):
        """Create ProfilingOptimizer instance."""
        return ProfilingOptimizer()
    
    def test_profiling_optimizer_creation(self, profiling_optimizer):
        """Test ProfilingOptimizer creation."""
        assert profiling_optimizer.name == "ProfilingOptimizer"
        assert hasattr(profiling_optimizer, 'can_optimize')
        assert hasattr(profiling_optimizer, 'get_optimization_capabilities')
    
    @pytest.mark.asyncio
    async def test_profiling_optimizer_placeholder_optimize(self, profiling_optimizer):
        """Test ProfilingOptimizer placeholder optimize method."""
        context = OptimizationContext(
            current_metrics={},
            target_metrics={},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        # This should raise NotImplementedError for now
        with pytest.raises(NotImplementedError):
            await profiling_optimizer.optimize(context)


class TestGPUOptimizer:
    """Test GPUOptimizer class."""
    
    @pytest.fixture
    def gpu_optimizer(self):
        """Create GPUOptimizer instance."""
        return GPUOptimizer()
    
    def test_gpu_optimizer_creation(self, gpu_optimizer):
        """Test GPUOptimizer creation."""
        assert gpu_optimizer.name == "GPUOptimizer"
        assert hasattr(gpu_optimizer, 'can_optimize')
        assert hasattr(gpu_optimizer, 'get_optimization_capabilities')
    
    @pytest.mark.asyncio
    async def test_gpu_optimizer_placeholder_optimize(self, gpu_optimizer):
        """Test GPUOptimizer placeholder optimize method."""
        context = OptimizationContext(
            current_metrics={},
            target_metrics={},
            constraints={},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        # This should raise NotImplementedError for now
        with pytest.raises(NotImplementedError):
            await gpu_optimizer.optimize(context)


if __name__ == "__main__":
    pytest.main([__file__])

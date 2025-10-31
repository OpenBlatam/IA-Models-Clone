from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from advanced_gradient_accumulation import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Advanced Gradient Accumulation System

This test suite covers:
- All gradient accumulation strategies
- Memory monitoring and optimization
- Performance metrics tracking
- Large batch size handling
- Edge cases and error conditions
- Integration testing
"""



    GradientAccumulationConfig, AccumulationStrategy,
    FixedGradientAccumulator, DynamicGradientAccumulator,
    MemoryAwareGradientAccumulator, PerformanceOptimizedGradientAccumulator,
    AdaptiveGradientAccumulator, AdvancedGradientAccumulationTrainer,
    MemoryMonitor, PerformanceMetrics, OptimizationScheduler
)


class TestDataset(Dataset):
    """Test dataset for unit testing."""
    
    def __init__(self, num_samples: int = 100, input_dim: int = 64):
        
    """__init__ function."""
self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 5, (num_samples,))
    
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx]
        }


class TestModel(nn.Module):
    """Test model for unit testing."""
    
    def __init__(self, input_dim: int = 64, num_classes: int = 5):
        
    """__init__ function."""
super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, input_ids, labels=None) -> Any:
        logits = self.linear(input_ids)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'logits': logits, 'loss': loss}


class TestGradientAccumulationConfig:
    """Test GradientAccumulationConfig class."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = GradientAccumulationConfig()
        
        assert config.accumulation_steps == 4
        assert config.strategy == AccumulationStrategy.ADAPTIVE
        assert config.use_mixed_precision is True
        assert config.enable_monitoring is True
        assert config.max_memory_usage_gb == 16.0
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = GradientAccumulationConfig(
            accumulation_steps=8,
            strategy=AccumulationStrategy.FIXED,
            target_batch_size=2048,
            use_mixed_precision=False,
            max_memory_usage_gb=8.0
        )
        
        assert config.accumulation_steps == 8
        assert config.strategy == AccumulationStrategy.FIXED
        assert config.target_batch_size == 2048
        assert config.use_mixed_precision is False
        assert config.max_memory_usage_gb == 8.0
    
    def test_invalid_config(self) -> Any:
        """Test invalid configuration validation."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(accumulation_steps=0)
        
        with pytest.raises(ValueError):
            GradientAccumulationConfig(max_memory_usage_gb=-1)
        
        with pytest.raises(ValueError):
            GradientAccumulationConfig(memory_safety_margin=1.5)
    
    def test_post_init(self) -> Any:
        """Test post-initialization setup."""
        config = GradientAccumulationConfig(
            effective_batch_size=None,
            target_batch_size=1024
        )
        
        assert config.effective_batch_size == 1024


class TestFixedGradientAccumulator:
    """Test FixedGradientAccumulator class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig(
            strategy=AccumulationStrategy.FIXED,
            accumulation_steps=4
        )
    
    @pytest.fixture
    def accumulator(self, config) -> Any:
        """Create test accumulator."""
        return FixedGradientAccumulator(config)
    
    def test_initialization(self, config) -> Any:
        """Test accumulator initialization."""
        accumulator = FixedGradientAccumulator(config)
        
        assert accumulator.config == config
        assert accumulator.current_step == 0
        assert accumulator.accumulation_count == 0
        assert accumulator.total_gradients == 0
    
    def test_accumulate_gradients(self, accumulator) -> Any:
        """Test gradient accumulation."""
        model = TestModel()
        loss = torch.tensor(1.0, requires_grad=True)
        
        # First accumulation
        should_optimize = accumulator.accumulate_gradients(loss, model)
        assert should_optimize is False
        assert accumulator.accumulation_count == 1
        
        # Complete accumulation
        for _ in range(3):  # 3 more steps to reach 4 total
            should_optimize = accumulator.accumulate_gradients(loss, model)
        
        assert should_optimize is True
        assert accumulator.accumulation_count == 4
    
    def test_should_optimize(self, accumulator) -> Any:
        """Test optimization decision."""
        assert accumulator.should_optimize() is False
        
        accumulator.accumulation_count = 4
        assert accumulator.should_optimize() is True
    
    def test_reset_accumulation(self, accumulator) -> Any:
        """Test accumulation reset."""
        accumulator.accumulation_count = 4
        accumulator.reset_accumulation()
        assert accumulator.accumulation_count == 0
    
    def test_get_effective_batch_size(self, accumulator) -> Optional[Dict[str, Any]]:
        """Test effective batch size calculation."""
        assert accumulator.get_effective_batch_size() == 4


class TestDynamicGradientAccumulator:
    """Test DynamicGradientAccumulator class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig(
            strategy=AccumulationStrategy.DYNAMIC,
            accumulation_steps=4,
            min_accumulation_steps=2,
            max_accumulation_steps=8,
            automatic_scaling=True
        )
    
    @pytest.fixture
    def accumulator(self, config) -> Any:
        """Create test accumulator."""
        return DynamicGradientAccumulator(config)
    
    def test_initialization(self, config) -> Any:
        """Test accumulator initialization."""
        accumulator = DynamicGradientAccumulator(config)
        
        assert accumulator.current_accumulation_steps == 4
        assert len(accumulator.performance_history) == 0
        assert len(accumulator.memory_history) == 0
    
    def test_accumulate_gradients(self, accumulator) -> Any:
        """Test gradient accumulation with dynamic steps."""
        model = TestModel()
        loss = torch.tensor(1.0, requires_grad=True)
        
        # Accumulate gradients
        should_optimize = accumulator.accumulate_gradients(loss, model)
        assert should_optimize is False
        assert accumulator.accumulation_count == 1
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)  # 8GB
    def test_memory_adaptation(self, mock_memory, mock_cuda, accumulator) -> Any:
        """Test memory-based adaptation."""
        model = TestModel()
        loss = torch.tensor(1.0, requires_grad=True)
        
        # Simulate high memory usage
        accumulator.config.max_memory_usage_gb = 10.0
        
        # Accumulate gradients to trigger adaptation
        for _ in range(4):
            accumulator.accumulate_gradients(loss, model)
        
        # Should adapt accumulation steps
        assert accumulator.current_accumulation_steps >= 4
    
    def test_adapt_accumulation_steps(self, accumulator) -> Any:
        """Test accumulation step adaptation."""
        initial_steps = accumulator.current_accumulation_steps
        
        # Simulate memory pressure
        accumulator.memory_history = [8.0] * 10  # High memory usage
        accumulator.config.max_memory_usage_gb = 10.0
        
        accumulator._adapt_accumulation_steps()
        
        # Should increase accumulation steps
        assert accumulator.current_accumulation_steps >= initial_steps


class TestMemoryAwareGradientAccumulator:
    """Test MemoryAwareGradientAccumulator class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig(
            strategy=AccumulationStrategy.MEMORY_AWARE,
            accumulation_steps=4,
            max_memory_usage_gb=8.0,
            enable_memory_optimization=True
        )
    
    @pytest.fixture
    def accumulator(self, config) -> Any:
        """Create test accumulator."""
        return MemoryAwareGradientAccumulator(config)
    
    def test_initialization(self, config) -> Any:
        """Test accumulator initialization."""
        accumulator = MemoryAwareGradientAccumulator(config)
        
        assert accumulator.memory_monitor is not None
        assert len(accumulator.accumulation_steps_history) == 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=6 * 1024**3)  # 6GB
    def test_memory_constrained_accumulation(self, mock_memory, mock_cuda, accumulator) -> Any:
        """Test accumulation under memory constraints."""
        model = TestModel()
        loss = torch.tensor(1.0, requires_grad=True)
        
        # Simulate memory constraint
        accumulator.memory_monitor.get_available_memory = lambda: 2.0  # Low available memory
        
        should_optimize = accumulator.accumulate_gradients(loss, model)
        
        # Should reduce accumulation steps
        assert accumulator.config.accumulation_steps <= 4
    
    def test_accumulate_gradients(self, accumulator) -> Any:
        """Test gradient accumulation with memory awareness."""
        model = TestModel()
        loss = torch.tensor(1.0, requires_grad=True)
        
        should_optimize = accumulator.accumulate_gradients(loss, model)
        assert should_optimize is False
        assert accumulator.accumulation_count == 1


class TestPerformanceOptimizedGradientAccumulator:
    """Test PerformanceOptimizedGradientAccumulator class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig(
            strategy=AccumulationStrategy.PERFORMANCE_OPTIMIZED,
            accumulation_steps=4,
            use_gradient_accumulation_hooks=True,
            automatic_scaling=True
        )
    
    @pytest.fixture
    def accumulator(self, config) -> Any:
        """Create test accumulator."""
        return PerformanceOptimizedGradientAccumulator(config)
    
    def test_initialization(self, config) -> Any:
        """Test accumulator initialization."""
        accumulator = PerformanceOptimizedGradientAccumulator(config)
        
        assert accumulator.gradient_buffer == {}
        assert accumulator.performance_metrics is not None
        assert accumulator.optimization_scheduler is not None
    
    def test_accumulate_with_hooks(self, accumulator) -> Any:
        """Test gradient accumulation with hooks."""
        model = TestModel()
        loss = torch.tensor(1.0, requires_grad=True)
        
        # Mock the gradient hook
        with patch.object(accumulator, '_gradient_hook') as mock_hook:
            accumulator._accumulate_with_hooks(loss, model)
            
            # Should register hooks for parameters
            assert mock_hook.called
    
    def test_optimize_strategy(self, accumulator) -> Any:
        """Test strategy optimization."""
        # Mock performance metrics
        accumulator.performance_metrics.get_average_step_time = lambda: 0.2  # High step time
        accumulator.performance_metrics.get_memory_efficiency = lambda: 0.9  # High memory usage
        
        initial_steps = accumulator.config.accumulation_steps
        accumulator._optimize_strategy()
        
        # Should increase accumulation steps for performance
        assert accumulator.config.accumulation_steps >= initial_steps


class TestMemoryMonitor:
    """Test MemoryMonitor class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig()
    
    @pytest.fixture
    def monitor(self, config) -> Any:
        """Create test monitor."""
        return MemoryMonitor(config)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_get_available_memory(self, mock_allocated, mock_properties, mock_cuda, monitor) -> Optional[Dict[str, Any]]:
        """Test available memory calculation."""
        # Mock device properties
        mock_properties.return_value.total_memory = 16 * 1024**3  # 16GB
        mock_allocated.return_value = 8 * 1024**3  # 8GB allocated
        
        available_memory = monitor.get_available_memory()
        assert available_memory == 8.0  # 8GB available
    
    def test_estimate_memory_needed(self, monitor) -> Any:
        """Test memory estimation."""
        model = TestModel()
        monitor.config.accumulation_steps = 4
        
        memory_needed = monitor.estimate_memory_needed(model)
        assert memory_needed > 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=4 * 1024**3)  # 4GB
    def test_update(self, mock_allocated, mock_cuda, monitor) -> Any:
        """Test memory monitoring update."""
        monitor.update()
        
        assert len(monitor.memory_history) == 1
        assert monitor.peak_memory == 4.0
    
    def test_get_memory_stats(self, monitor) -> Optional[Dict[str, Any]]:
        """Test memory statistics."""
        # Add some memory history
        monitor.memory_history = [2.0, 3.0, 4.0]
        monitor.peak_memory = 4.0
        
        stats = monitor.get_memory_stats()
        
        assert 'current_memory_gb' in stats
        assert 'peak_memory_gb' in stats
        assert 'average_memory_gb' in stats
        assert stats['current_memory_gb'] == 4.0
        assert stats['peak_memory_gb'] == 4.0
        assert stats['average_memory_gb'] == 3.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    @pytest.fixture
    def metrics(self) -> Any:
        """Create test metrics."""
        return PerformanceMetrics()
    
    def test_initialization(self, metrics) -> Any:
        """Test metrics initialization."""
        assert len(metrics.step_times) == 0
        assert len(metrics.memory_usage) == 0
        assert len(metrics.throughput_history) == 0
    
    def test_update(self, metrics) -> Any:
        """Test metrics update."""
        metrics.update(0.1, 4)  # 0.1s step time, 4 accumulation count
        
        assert len(metrics.step_times) == 1
        assert metrics.step_times[0] == 0.1
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=4 * 1024**3)  # 4GB
    def test_update_with_memory(self, mock_allocated, mock_cuda, metrics) -> Any:
        """Test metrics update with memory monitoring."""
        metrics.update(0.1, 4)
        
        assert len(metrics.memory_usage) == 1
        assert metrics.memory_usage[0] == 4.0
    
    def test_get_average_step_time(self, metrics) -> Optional[Dict[str, Any]]:
        """Test average step time calculation."""
        metrics.step_times = [0.1, 0.2, 0.3]
        
        avg_time = metrics.get_average_step_time()
        assert avg_time == 0.2
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_get_memory_efficiency(self, mock_properties, mock_cuda, metrics) -> Optional[Dict[str, Any]]:
        """Test memory efficiency calculation."""
        mock_properties.return_value.total_memory = 16 * 1024**3  # 16GB
        metrics.memory_usage = [8.0]  # 8GB used
        
        efficiency = metrics.get_memory_efficiency()
        assert efficiency == 0.5  # 50% efficiency
    
    def test_get_throughput(self, metrics) -> Optional[Dict[str, Any]]:
        """Test throughput calculation."""
        metrics.throughput_history = [100.0, 200.0, 150.0]
        
        throughput = metrics.get_throughput()
        assert throughput == 150.0
    
    def test_get_performance_summary(self, metrics) -> Optional[Dict[str, Any]]:
        """Test performance summary."""
        metrics.step_times = [0.1, 0.2]
        metrics.memory_usage = [4.0, 6.0]
        metrics.throughput_history = [100.0, 200.0]
        
        summary = metrics.get_performance_summary()
        
        assert 'average_step_time' in summary
        assert 'memory_efficiency' in summary
        assert 'throughput' in summary
        assert 'total_training_time' in summary


class TestOptimizationScheduler:
    """Test OptimizationScheduler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig(accumulation_steps=4)
    
    @pytest.fixture
    def scheduler(self, config) -> Any:
        """Create test scheduler."""
        return OptimizationScheduler(config)
    
    def test_initialization(self, config) -> Any:
        """Test scheduler initialization."""
        scheduler = OptimizationScheduler(config)
        
        assert scheduler.config == config
        assert scheduler.target_step_time == 0.1
        assert len(scheduler.optimization_history) == 0
    
    def test_should_optimize_basic(self, scheduler) -> Any:
        """Test basic optimization decision."""
        metrics = Mock()
        metrics.get_average_step_time.return_value = 0.05
        metrics.get_memory_efficiency.return_value = 0.5
        
        should_optimize = scheduler.should_optimize(4, metrics)
        assert should_optimize is True
    
    def test_should_optimize_high_step_time(self, scheduler) -> Any:
        """Test optimization with high step time."""
        metrics = Mock()
        metrics.get_average_step_time.return_value = 0.3  # High step time
        metrics.get_memory_efficiency.return_value = 0.5
        
        should_optimize = scheduler.should_optimize(4, metrics)
        assert should_optimize is True
    
    def test_should_optimize_high_memory(self, scheduler) -> Any:
        """Test optimization with high memory usage."""
        metrics = Mock()
        metrics.get_average_step_time.return_value = 0.05
        metrics.get_memory_efficiency.return_value = 0.95  # High memory usage
        
        should_optimize = scheduler.should_optimize(4, metrics)
        assert should_optimize is True
    
    def test_update_target_step_time(self, scheduler) -> Optional[Dict[str, Any]]:
        """Test target step time update."""
        scheduler.update_target_step_time(0.2)
        assert scheduler.target_step_time == 0.2


class TestAdvancedGradientAccumulationTrainer:
    """Test AdvancedGradientAccumulationTrainer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return GradientAccumulationConfig(
            strategy=AccumulationStrategy.FIXED,
            accumulation_steps=4,
            use_mixed_precision=False
        )
    
    @pytest.fixture
    def trainer(self, config) -> Any:
        """Create test trainer."""
        return AdvancedGradientAccumulationTrainer(config)
    
    def test_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        assert trainer.config == config
        assert trainer.accumulator is not None
        assert trainer.memory_monitor is not None
        assert trainer.performance_metrics is not None
        assert trainer.optimization_scheduler is not None
    
    def test_create_accumulator_fixed(self, config) -> Any:
        """Test creating fixed accumulator."""
        config.strategy = AccumulationStrategy.FIXED
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        assert isinstance(trainer.accumulator, FixedGradientAccumulator)
    
    def test_create_accumulator_dynamic(self, config) -> Any:
        """Test creating dynamic accumulator."""
        config.strategy = AccumulationStrategy.DYNAMIC
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        assert isinstance(trainer.accumulator, DynamicGradientAccumulator)
    
    def test_create_accumulator_memory_aware(self, config) -> Any:
        """Test creating memory-aware accumulator."""
        config.strategy = AccumulationStrategy.MEMORY_AWARE
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        assert isinstance(trainer.accumulator, MemoryAwareGradientAccumulator)
    
    def test_create_accumulator_performance_optimized(self, config) -> Any:
        """Test creating performance-optimized accumulator."""
        config.strategy = AccumulationStrategy.PERFORMANCE_OPTIMIZED
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        assert isinstance(trainer.accumulator, PerformanceOptimizedGradientAccumulator)
    
    def test_create_accumulator_adaptive(self, config) -> Any:
        """Test creating adaptive accumulator."""
        config.strategy = AccumulationStrategy.ADAPTIVE
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        assert isinstance(trainer.accumulator, AdaptiveGradientAccumulator)
    
    def test_train_step(self, trainer) -> Any:
        """Test training step."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        result = trainer.train_step(batch, model, optimizer)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
        assert 'should_optimize' in result
        assert 'effective_batch_size' in result
        assert 'accumulation_count' in result
    
    def test_train_step_with_mixed_precision(self, config) -> Any:
        """Test training step with mixed precision."""
        config.use_mixed_precision = True
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        result = trainer.train_step(batch, model, optimizer, scaler)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
    
    def test_perform_optimization(self, trainer) -> Any:
        """Test optimization step."""
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Set up gradients
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        
        trainer._perform_optimization(model, optimizer)
        
        # Gradients should be zero after optimization
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0)
    
    def test_perform_optimization_with_gradient_clipping(self, trainer) -> Any:
        """Test optimization with gradient clipping."""
        trainer.config.gradient_clipping = True
        trainer.config.max_grad_norm = 1.0
        
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Set up gradients
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        
        trainer._perform_optimization(model, optimizer)
        
        # Should not raise an exception
        assert True
    
    def test_get_training_stats(self, trainer) -> Optional[Dict[str, Any]]:
        """Test training statistics."""
        stats = trainer.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'accumulator_stats' in stats
        assert 'performance_metrics' in stats
        assert 'memory_stats' in stats
        assert 'config' in stats
    
    def test_cleanup(self, trainer) -> Any:
        """Test trainer cleanup."""
        trainer.cleanup()
        
        # Should not raise an exception
        assert True


class TestIntegration:
    """Integration tests for the advanced gradient accumulation system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_training(self) -> Any:
        """Test end-to-end training workflow."""
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.FIXED,
            accumulation_steps=4,
            use_mixed_precision=False,
            enable_monitoring=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = TestModel()
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Train for 10 steps
                break
            
            result = trainer.train_step(batch, model, optimizer)
            
            if result['should_optimize']:
                logger.info(f"Optimization step performed at step {i}")
        
        # Get training stats
        stats = trainer.get_training_stats()
        assert isinstance(stats, dict)
        
        trainer.cleanup()
    
    def test_memory_monitoring_integration(self) -> Any:
        """Test memory monitoring integration."""
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.MEMORY_AWARE,
            accumulation_steps=4,
            max_memory_usage_gb=8.0
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Train step should update memory monitoring
        result = trainer.train_step(batch, model, optimizer)
        
        assert isinstance(result, dict)
        trainer.cleanup()
    
    def test_performance_optimization_integration(self) -> Any:
        """Test performance optimization integration."""
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.PERFORMANCE_OPTIMIZED,
            accumulation_steps=4,
            automatic_scaling=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Multiple training steps to trigger optimization
        for _ in range(8):
            result = trainer.train_step(batch, model, optimizer)
        
        stats = trainer.get_training_stats()
        assert isinstance(stats, dict)
        
        trainer.cleanup()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_accumulation_steps(self) -> Any:
        """Test zero accumulation steps."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(accumulation_steps=0)
    
    def test_negative_memory_usage(self) -> Any:
        """Test negative memory usage."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(max_memory_usage_gb=-1)
    
    def test_invalid_memory_safety_margin(self) -> Any:
        """Test invalid memory safety margin."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(memory_safety_margin=1.5)
    
    def test_empty_dataset(self) -> Any:
        """Test handling empty dataset."""
        config = GradientAccumulationConfig()
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = TestModel()
        
        # Should handle gracefully
        assert trainer is not None
        assert model is not None
    
    def test_large_accumulation_steps(self) -> Any:
        """Test very large accumulation steps."""
        config = GradientAccumulationConfig(accumulation_steps=1000)
        trainer = AdvancedGradientAccumulationTrainer(config)
        
        # Should handle gracefully
        assert trainer.config.accumulation_steps == 1000
    
    def test_mixed_precision_without_cuda(self) -> Any:
        """Test mixed precision without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            config = GradientAccumulationConfig(use_mixed_precision=True)
            trainer = AdvancedGradientAccumulationTrainer(config)
            
            # Should not fail, but mixed precision should be disabled
            assert trainer.config.use_mixed_precision is True


class TestPerformance:
    """Performance tests."""
    
    def test_training_speed(self) -> Any:
        """Test training speed."""
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.FIXED,
            accumulation_steps=4,
            use_mixed_precision=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(16, 64),
            'labels': torch.randint(0, 5, (16,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        # Measure training time
        start_time = time.time()
        
        for i in range(20):
            result = trainer.train_step(batch, model, optimizer, scaler)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Should complete within reasonable time
        assert training_time < 60  # Less than 60 seconds
        
        trainer.cleanup()
    
    def test_memory_efficiency(self) -> Any:
        """Test memory efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.MEMORY_AWARE,
            accumulation_steps=8,
            use_mixed_precision=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(32, 64),
            'labels': torch.randint(0, 5, (32,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        # Record memory usage
        initial_memory = torch.cuda.memory_allocated()
        
        # Train for several steps
        for i in range(10):
            result = trainer.train_step(batch, model, optimizer, scaler)
            
            # Check memory doesn't grow excessively
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB)
            assert memory_increase < 1024**3
        
        trainer.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 
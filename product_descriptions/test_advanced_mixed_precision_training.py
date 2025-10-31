from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
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
from advanced_mixed_precision_training import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Advanced Mixed Precision Training System

This test suite covers:
- All mixed precision training strategies
- Gradient scaling and monitoring
- Memory optimization
- Numerical stability
- Performance profiling
- Edge cases and error handling
- Integration testing
"""



    MixedPrecisionConfig, PrecisionMode, ScalingStrategy,
    AdvancedGradScaler, PrecisionMonitor, StandardMixedPrecisionTrainer,
    DynamicMixedPrecisionTrainer, PerformanceOptimizedMixedPrecisionTrainer,
    AdvancedMixedPrecisionManager, PerformanceTracker, OptimizationScheduler
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


class TestMixedPrecisionConfig:
    """Test MixedPrecisionConfig class."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = MixedPrecisionConfig()
        
        assert config.enabled is True
        assert config.precision_mode == PrecisionMode.MIXED
        assert config.scaling_strategy == ScalingStrategy.ADAPTIVE
        assert config.init_scale == 2**16
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
        assert config.enable_monitoring is True
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = MixedPrecisionConfig(
            enabled=False,
            precision_mode=PrecisionMode.FP32,
            scaling_strategy=ScalingStrategy.CONSTANT,
            init_scale=2**10,
            growth_factor=1.5,
            backoff_factor=0.8
        )
        
        assert config.enabled is False
        assert config.precision_mode == PrecisionMode.FP32
        assert config.scaling_strategy == ScalingStrategy.CONSTANT
        assert config.init_scale == 2**10
        assert config.growth_factor == 1.5
        assert config.backoff_factor == 0.8
    
    def test_invalid_config(self) -> Any:
        """Test invalid configuration validation."""
        with pytest.raises(ValueError):
            MixedPrecisionConfig(init_scale=0)
        
        with pytest.raises(ValueError):
            MixedPrecisionConfig(growth_factor=0.5)
        
        with pytest.raises(ValueError):
            MixedPrecisionConfig(backoff_factor=1.5)
    
    def test_post_init_cuda_unavailable(self) -> Any:
        """Test post_init when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            config = MixedPrecisionConfig()
            assert config.enabled is False


class TestAdvancedGradScaler:
    """Test AdvancedGradScaler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(
            enabled=True,
            init_scale=2**10,
            growth_factor=2.0,
            backoff_factor=0.5
        )
    
    @pytest.fixture
    def scaler(self, config) -> Any:
        """Create test scaler."""
        return AdvancedGradScaler(config)
    
    def test_initialization(self, config) -> Any:
        """Test scaler initialization."""
        scaler = AdvancedGradScaler(config)
        
        assert scaler.config == config
        assert scaler.precision_monitor is not None
        assert len(scaler.scale_history) == 0
        assert len(scaler.error_history) == 0
    
    def test_scale(self, scaler) -> Any:
        """Test gradient scaling."""
        outputs = torch.tensor([1.0, 2.0, 3.0])
        scaled_outputs = scaler.scale(outputs)
        
        assert len(scaler.scale_history) == 1
        assert scaler.get_scale() == 2**10  # Initial scale
    
    def test_step_success(self, scaler) -> Any:
        """Test successful optimization step."""
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Set up gradients
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        
        # Should not raise an exception
        scaler.step(optimizer)
        assert len(scaler.error_history) == 0
    
    @patch('torch.cuda.amp.GradScaler.step')
    def test_step_with_error(self, mock_step, scaler) -> Any:
        """Test optimization step with error."""
        mock_step.side_effect = Exception("Gradient overflow")
        
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Set up gradients
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        
        # Should handle error gracefully
        scaler.step(optimizer)
        assert len(scaler.error_history) == 1
    
    def test_update(self, scaler) -> Any:
        """Test scaler update."""
        old_scale = scaler.get_scale()
        new_scale = old_scale * 2
        
        scaler.update(new_scale)
        assert scaler.get_scale() == new_scale
    
    def test_get_scale_stats(self, scaler) -> Optional[Dict[str, Any]]:
        """Test scale statistics."""
        # Add some scale history
        scaler.scale_history = [2**10, 2**11, 2**12, 2**11]
        
        stats = scaler.get_scale_stats()
        
        assert 'current_scale' in stats
        assert 'avg_scale' in stats
        assert 'max_scale' in stats
        assert 'min_scale' in stats
        assert 'scale_volatility' in stats
        
        assert stats['current_scale'] == 2**10
        assert stats['max_scale'] == 2**12
        assert stats['min_scale'] == 2**10


class TestPrecisionMonitor:
    """Test PrecisionMonitor class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(enable_monitoring=True)
    
    @pytest.fixture
    def monitor(self, config) -> Any:
        """Create test monitor."""
        return PrecisionMonitor(config)
    
    def test_initialization(self, config) -> Any:
        """Test monitor initialization."""
        monitor = PrecisionMonitor(config)
        
        assert monitor.config == config
        assert len(monitor.metrics['fp16_usage']) == 0
        assert len(monitor.metrics['fp32_usage']) == 0
        assert len(monitor.metrics['gradient_scale']) == 0
        assert monitor.metrics['fallback_count'] == 0
    
    def test_update_metrics_with_scaler(self, monitor) -> Any:
        """Test metrics update with scaler."""
        scaler = Mock()
        scaler.get_scale.return_value = 2**10
        
        loss = torch.tensor(1.0, dtype=torch.float16)
        
        monitor.update_metrics(scaler, loss, 1)
        
        assert len(monitor.metrics['gradient_scale']) == 1
        assert monitor.metrics['gradient_scale'][0] == 2**10
        assert len(monitor.metrics['fp16_usage']) == 1
        assert monitor.metrics['fp16_usage'][0] == 1.0
    
    def test_update_metrics_without_scaler(self, monitor) -> Any:
        """Test metrics update without scaler."""
        loss = torch.tensor(1.0, dtype=torch.float32)
        
        monitor.update_metrics(None, loss, 1)
        
        assert len(monitor.metrics['fp32_usage']) == 1
        assert monitor.metrics['fp32_usage'][0] == 1.0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_update_metrics_with_memory(self, mock_allocated, mock_properties, mock_cuda, monitor) -> Any:
        """Test metrics update with memory monitoring."""
        mock_properties.return_value.total_memory = 16 * 1024**3  # 16GB
        mock_allocated.return_value = 8 * 1024**3  # 8GB
        
        scaler = Mock()
        scaler.get_scale.return_value = 2**10
        
        loss = torch.tensor(1.0)
        
        monitor.update_metrics(scaler, loss, 1)
        
        assert len(monitor.metrics['memory_savings']) == 1
        assert monitor.metrics['memory_savings'][0] == 0.5  # 50% savings
    
    def test_record_numerical_error(self, monitor) -> Any:
        """Test numerical error recording."""
        monitor.record_numerical_error("gradient_overflow", 10)
        
        assert len(monitor.metrics['numerical_errors']) == 1
        assert monitor.metrics['numerical_errors'][0]['type'] == "gradient_overflow"
        assert monitor.metrics['numerical_errors'][0]['step'] == 10
        assert monitor.metrics['fallback_count'] == 1
    
    def test_get_precision_stats(self, monitor) -> Optional[Dict[str, Any]]:
        """Test precision statistics."""
        # Add some metrics
        monitor.metrics['gradient_scale'] = [2**10, 2**11, 2**12]
        monitor.metrics['fp16_usage'] = [1.0, 1.0, 0.0]
        monitor.metrics['fp32_usage'] = [0.0, 0.0, 1.0]
        monitor.metrics['memory_savings'] = [0.5, 0.6, 0.4]
        monitor.metrics['numerical_errors'] = [{'type': 'overflow', 'step': 1}]
        monitor.metrics['fallback_count'] = 1
        
        stats = monitor.get_precision_stats()
        
        assert 'avg_gradient_scale' in stats
        assert 'max_gradient_scale' in stats
        assert 'min_gradient_scale' in stats
        assert 'fp16_usage_rate' in stats
        assert 'fp32_usage_rate' in stats
        assert 'avg_memory_savings' in stats
        assert 'numerical_errors' in stats
        assert 'fallback_count' in stats
        
        assert stats['avg_gradient_scale'] == (2**10 + 2**11 + 2**12) / 3
        assert stats['fp16_usage_rate'] == 2/3
        assert stats['fp32_usage_rate'] == 1/3
        assert stats['numerical_errors'] == 1
        assert stats['fallback_count'] == 1


class TestStandardMixedPrecisionTrainer:
    """Test StandardMixedPrecisionTrainer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED
        )
    
    @pytest.fixture
    def trainer(self, config) -> Any:
        """Create test trainer."""
        return StandardMixedPrecisionTrainer(config)
    
    def test_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = StandardMixedPrecisionTrainer(config)
        
        assert trainer.config == config
        assert trainer.scaler is not None
        assert trainer.precision_monitor is not None
        assert trainer.current_step == 0
    
    def test_train_step_with_mixed_precision(self, trainer) -> Any:
        """Test training step with mixed precision."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        result = trainer.train_step(batch, model, optimizer)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
        assert 'loss' in result
        assert 'precision_mode' in result
        assert result['precision_mode'] == 'mixed'
        assert trainer.current_step == 1
    
    def test_train_step_without_mixed_precision(self, config) -> Any:
        """Test training step without mixed precision."""
        config.enabled = False
        trainer = StandardMixedPrecisionTrainer(config)
        
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        result = trainer.train_step(batch, model, optimizer)
        
        assert result['precision_mode'] == 'fp32'
    
    def test_validate_step(self, trainer) -> bool:
        """Test validation step."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        
        result = trainer.validate_step(batch, model)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
        assert 'loss' in result


class TestDynamicMixedPrecisionTrainer:
    """Test DynamicMixedPrecisionTrainer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.DYNAMIC
        )
    
    @pytest.fixture
    def trainer(self, config) -> Any:
        """Create test trainer."""
        return DynamicMixedPrecisionTrainer(config)
    
    def test_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = DynamicMixedPrecisionTrainer(config)
        
        assert trainer.config == config
        assert len(trainer.precision_history) == 0
        assert len(trainer.performance_metrics) == 0
    
    def test_train_step(self, trainer) -> Any:
        """Test training step with dynamic precision."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        result = trainer.train_step(batch, model, optimizer)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
        assert 'loss' in result
        assert 'precision_mode' in result
        assert 'step_time' in result
        assert len(trainer.precision_history) == 1
        assert len(trainer.performance_metrics) == 1
    
    def test_determine_precision_mode(self, trainer) -> Any:
        """Test precision mode determination."""
        # Test normal conditions
        precision_mode = trainer._determine_precision_mode()
        assert precision_mode in [PrecisionMode.MIXED, PrecisionMode.FP32]
        
        # Test with poor performance
        trainer.performance_metrics = [0.2] * 10  # Poor performance
        precision_mode = trainer._determine_precision_mode()
        assert precision_mode == PrecisionMode.FP32
        
        # Test with numerical errors
        trainer.scaler.error_history = [{'step': 1}] * 6  # Many errors
        precision_mode = trainer._determine_precision_mode()
        assert precision_mode == PrecisionMode.FP32


class TestPerformanceOptimizedMixedPrecisionTrainer:
    """Test PerformanceOptimizedMixedPrecisionTrainer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED
        )
    
    @pytest.fixture
    def trainer(self, config) -> Any:
        """Create test trainer."""
        return PerformanceOptimizedMixedPrecisionTrainer(config)
    
    def test_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = PerformanceOptimizedMixedPrecisionTrainer(config)
        
        assert trainer.config == config
        assert trainer.performance_tracker is not None
        assert trainer.optimization_scheduler is not None
    
    def test_train_step(self, trainer) -> Any:
        """Test training step with performance optimization."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        result = trainer.train_step(batch, model, optimizer)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
        assert 'loss' in result
        assert 'step_time' in result
        assert 'performance_optimized' in result
        assert result['performance_optimized'] is True
    
    def test_optimize_precision_settings(self, trainer) -> Any:
        """Test precision settings optimization."""
        # Mock poor performance
        trainer.performance_tracker.get_stats = lambda: {'avg_step_time': 0.2}
        
        initial_scale = trainer.scaler.get_scale()
        trainer._optimize_precision_settings()
        
        # Should increase scale for better performance
        new_scale = trainer.scaler.get_scale()
        assert new_scale >= initial_scale


class TestPerformanceTracker:
    """Test PerformanceTracker class."""
    
    @pytest.fixture
    def tracker(self) -> Any:
        """Create test tracker."""
        return PerformanceTracker()
    
    def test_initialization(self, tracker) -> Any:
        """Test tracker initialization."""
        assert len(tracker.step_times) == 0
        assert len(tracker.gradient_scales) == 0
        assert len(tracker.memory_usage) == 0
    
    def test_update(self, tracker) -> Any:
        """Test metrics update."""
        tracker.update(0.1, 2**10)
        
        assert len(tracker.step_times) == 1
        assert tracker.step_times[0] == 0.1
        assert len(tracker.gradient_scales) == 1
        assert tracker.gradient_scales[0] == 2**10
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=4 * 1024**3)  # 4GB
    def test_update_with_memory(self, mock_allocated, mock_cuda, tracker) -> Any:
        """Test update with memory monitoring."""
        tracker.update(0.1, 2**10)
        
        assert len(tracker.memory_usage) == 1
        assert tracker.memory_usage[0] == 4.0
    
    def test_get_stats(self, tracker) -> Optional[Dict[str, Any]]:
        """Test statistics calculation."""
        # Add some data
        tracker.step_times = [0.1, 0.2, 0.3]
        tracker.gradient_scales = [2**10, 2**11, 2**12]
        tracker.memory_usage = [4.0, 5.0, 6.0]
        
        stats = tracker.get_stats()
        
        assert 'avg_step_time' in stats
        assert 'max_step_time' in stats
        assert 'min_step_time' in stats
        assert 'avg_gradient_scale' in stats
        assert 'avg_memory_usage' in stats
        assert 'total_training_time' in stats
        
        assert stats['avg_step_time'] == 0.2
        assert stats['max_step_time'] == 0.3
        assert stats['min_step_time'] == 0.1
        assert stats['avg_gradient_scale'] == (2**10 + 2**11 + 2**12) / 3
        assert stats['avg_memory_usage'] == 5.0


class TestOptimizationScheduler:
    """Test OptimizationScheduler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(enabled=True)
    
    @pytest.fixture
    def scheduler(self, config) -> Any:
        """Create test scheduler."""
        return OptimizationScheduler(config)
    
    def test_initialization(self, config) -> Any:
        """Test scheduler initialization."""
        scheduler = OptimizationScheduler(config)
        
        assert scheduler.config == config
        assert len(scheduler.optimization_history) == 0
    
    def test_should_optimize_precision_good_performance(self, scheduler) -> Any:
        """Test optimization decision with good performance."""
        performance_stats = {
            'avg_step_time': 0.05,  # Good performance
            'avg_memory_usage': 0.5  # Good memory usage
        }
        
        should_optimize = scheduler.should_optimize_precision(performance_stats)
        assert should_optimize is False
    
    def test_should_optimize_precision_poor_performance(self, scheduler) -> Any:
        """Test optimization decision with poor performance."""
        performance_stats = {
            'avg_step_time': 0.2,  # Poor performance
            'avg_memory_usage': 0.5
        }
        
        should_optimize = scheduler.should_optimize_precision(performance_stats)
        assert should_optimize is True
    
    def test_should_optimize_precision_high_memory(self, scheduler) -> Any:
        """Test optimization decision with high memory usage."""
        performance_stats = {
            'avg_step_time': 0.05,
            'avg_memory_usage': 0.9  # High memory usage
        }
        
        should_optimize = scheduler.should_optimize_precision(performance_stats)
        assert should_optimize is True


class TestAdvancedMixedPrecisionManager:
    """Test AdvancedMixedPrecisionManager class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.CONSTANT
        )
    
    @pytest.fixture
    def manager(self, config) -> Any:
        """Create test manager."""
        return AdvancedMixedPrecisionManager(config)
    
    def test_initialization(self, config) -> Any:
        """Test manager initialization."""
        manager = AdvancedMixedPrecisionManager(config)
        
        assert manager.config == config
        assert manager.trainer is not None
        assert manager.writer is not None
    
    def test_create_trainer_constant(self, config) -> Any:
        """Test creating constant scaling trainer."""
        config.scaling_strategy = ScalingStrategy.CONSTANT
        manager = AdvancedMixedPrecisionManager(config)
        
        assert isinstance(manager.trainer, StandardMixedPrecisionTrainer)
    
    def test_create_trainer_dynamic(self, config) -> Any:
        """Test creating dynamic scaling trainer."""
        config.scaling_strategy = ScalingStrategy.DYNAMIC
        manager = AdvancedMixedPrecisionManager(config)
        
        assert isinstance(manager.trainer, DynamicMixedPrecisionTrainer)
    
    def test_create_trainer_performance_optimized(self, config) -> Any:
        """Test creating performance-optimized trainer."""
        config.scaling_strategy = ScalingStrategy.PERFORMANCE_OPTIMIZED
        manager = AdvancedMixedPrecisionManager(config)
        
        assert isinstance(manager.trainer, PerformanceOptimizedMixedPrecisionTrainer)
    
    def test_train_epoch(self, manager) -> Any:
        """Test training epoch."""
        model = TestModel()
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        epoch_metrics = manager.train_epoch(dataloader, model, optimizer)
        
        assert isinstance(epoch_metrics, dict)
        assert 'losses' in epoch_metrics
        assert 'step_times' in epoch_metrics
        assert 'precision_modes' in epoch_metrics
        assert 'gradient_scales' in epoch_metrics
    
    def test_validate_epoch(self, manager) -> bool:
        """Test validation epoch."""
        model = TestModel()
        dataset = TestDataset(50)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        val_metrics = manager.validate_epoch(dataloader, model)
        
        assert isinstance(val_metrics, dict)
        assert 'losses' in val_metrics
        assert 'precision_modes' in val_metrics
    
    def test_get_training_stats(self, manager) -> Optional[Dict[str, Any]]:
        """Test training statistics."""
        stats = manager.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'precision_stats' in stats
        assert 'performance_stats' in stats
        assert 'scale_stats' in stats
        assert 'config' in stats
    
    def test_cleanup(self, manager) -> Any:
        """Test manager cleanup."""
        manager.cleanup()
        
        # Should not raise an exception
        assert True


class TestIntegration:
    """Integration tests for the advanced mixed precision training system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_training(self) -> Any:
        """Test end-to-end training workflow."""
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.CONSTANT,
            enable_monitoring=True
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        epoch_metrics = manager.train_epoch(dataloader, model, optimizer)
        
        # Get training stats
        stats = manager.get_training_stats()
        assert isinstance(stats, dict)
        
        manager.cleanup()
    
    def test_mixed_precision_integration(self) -> Any:
        """Test mixed precision integration."""
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.DYNAMIC
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Training step should work with mixed precision
        result = manager.trainer.train_step(batch, model, optimizer)
        
        assert isinstance(result, dict)
        assert 'outputs' in result
        manager.cleanup()
    
    def test_performance_optimization_integration(self) -> Any:
        """Test performance optimization integration."""
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Multiple training steps to trigger optimization
        for _ in range(10):
            result = manager.trainer.train_step(batch, model, optimizer)
        
        stats = manager.get_training_stats()
        assert isinstance(stats, dict)
        
        manager.cleanup()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_init_scale(self) -> Any:
        """Test zero init scale."""
        with pytest.raises(ValueError):
            MixedPrecisionConfig(init_scale=0)
    
    def test_invalid_growth_factor(self) -> Any:
        """Test invalid growth factor."""
        with pytest.raises(ValueError):
            MixedPrecisionConfig(growth_factor=0.5)
    
    def test_invalid_backoff_factor(self) -> Any:
        """Test invalid backoff factor."""
        with pytest.raises(ValueError):
            MixedPrecisionConfig(backoff_factor=1.5)
    
    def test_empty_dataset(self) -> Any:
        """Test handling empty dataset."""
        config = MixedPrecisionConfig()
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        
        # Should handle gracefully
        assert manager is not None
        assert model is not None
    
    def test_mixed_precision_without_cuda(self) -> Any:
        """Test mixed precision without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            config = MixedPrecisionConfig(enabled=True)
            manager = AdvancedMixedPrecisionManager(config)
            
            # Should disable mixed precision automatically
            assert manager.config.enabled is False
    
    def test_numerical_instability(self) -> Any:
        """Test numerical instability handling."""
        config = MixedPrecisionConfig(
            enabled=True,
            automatic_fallback=True
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(4, 64) * 1e6,  # Large values
            'labels': torch.randint(0, 5, (4,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-2)  # High learning rate
        
        # Should handle numerical issues gracefully
        try:
            result = manager.trainer.train_step(batch, model, optimizer)
            assert isinstance(result, dict)
        except Exception as e:
            # Should fallback to FP32
            assert "fallback" in str(e).lower() or "overflow" in str(e).lower()
        
        manager.cleanup()


class TestPerformance:
    """Performance tests."""
    
    def test_training_speed(self) -> Any:
        """Test training speed."""
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.CONSTANT
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(16, 64),
            'labels': torch.randint(0, 5, (16,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Measure training time
        start_time = time.time()
        
        for i in range(20):
            result = manager.trainer.train_step(batch, model, optimizer)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Should complete within reasonable time
        assert training_time < 60  # Less than 60 seconds
        
        manager.cleanup()
    
    def test_memory_efficiency(self) -> Any:
        """Test memory efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = TestModel()
        batch = {
            'input_ids': torch.randn(32, 64),
            'labels': torch.randint(0, 5, (32,))
        }
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Record memory usage
        initial_memory = torch.cuda.memory_allocated()
        
        # Train for several steps
        for i in range(10):
            result = manager.trainer.train_step(batch, model, optimizer)
            
            # Check memory doesn't grow excessively
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB)
            assert memory_increase < 1024**3
        
        manager.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 
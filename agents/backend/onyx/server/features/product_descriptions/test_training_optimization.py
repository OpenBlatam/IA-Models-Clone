from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training_optimization import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Training Optimization System

This test suite covers:
- Early stopping functionality with different strategies
- Learning rate scheduling with various algorithms
- Gradient optimization and clipping
- Training monitoring and logging
- Checkpoint management
- Performance optimization
- Real-world scenarios
- Error handling and edge cases
"""



    EarlyStoppingConfig, LRSchedulerConfig, TrainingOptimizationConfig,
    EarlyStoppingMode, LRSchedulerType, EarlyStopping, LRSchedulerFactory,
    GradientOptimizer, TrainingMonitor, OptimizedTrainer, create_optimized_trainer,
    load_checkpoint
)


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    @pytest.fixture
    def simple_model(self) -> Any:
        """Create a simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        model.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return model
    
    @pytest.fixture
    def early_stopping_config(self) -> Any:
        """Create early stopping configuration."""
        return EarlyStoppingConfig(
            mode=EarlyStoppingMode.MIN,
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=True
        )
    
    def test_early_stopping_initialization(self, early_stopping_config) -> Any:
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(early_stopping_config)
        
        assert early_stopping.config == early_stopping_config
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert early_stopping.best_epoch == 0
        assert early_stopping.best_weights is None
    
    def test_early_stopping_min_mode_improvement(self, early_stopping_config, simple_model) -> Any:
        """Test early stopping with min mode and improvement."""
        early_stopping = EarlyStopping(early_stopping_config)
        
        # First call - should not stop
        result = early_stopping(0, 0.5, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.5
        assert early_stopping.counter == 0
        
        # Improvement - should not stop
        result = early_stopping(1, 0.3, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.3
        assert early_stopping.counter == 0
        
        # No improvement - should not stop yet
        result = early_stopping(2, 0.4, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.3
        assert early_stopping.counter == 1
    
    def test_early_stopping_max_mode_improvement(self, simple_model) -> Any:
        """Test early stopping with max mode and improvement."""
        config = EarlyStoppingConfig(
            mode=EarlyStoppingMode.MAX,
            patience=3,
            monitor="val_accuracy"
        )
        early_stopping = EarlyStopping(config)
        
        # First call - should not stop
        result = early_stopping(0, 0.5, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.5
        
        # Improvement - should not stop
        result = early_stopping(1, 0.7, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.7
        
        # No improvement - should not stop yet
        result = early_stopping(2, 0.6, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.7
        assert early_stopping.counter == 1
    
    def test_early_stopping_patience_exceeded(self, early_stopping_config, simple_model) -> Any:
        """Test early stopping when patience is exceeded."""
        early_stopping = EarlyStopping(early_stopping_config)
        
        # Set initial best score
        early_stopping.best_score = 0.5
        
        # Exceed patience
        for epoch in range(1, 7):
            result = early_stopping(epoch, 0.6, simple_model)
            if epoch < 6:
                assert result is False
            else:
                assert result is True
    
    def test_early_stopping_min_epochs(self, simple_model) -> Any:
        """Test early stopping with minimum epochs."""
        config = EarlyStoppingConfig(
            patience=3,
            min_epochs=5
        )
        early_stopping = EarlyStopping(config)
        
        # Should not stop before min_epochs
        for epoch in range(5):
            result = early_stopping(epoch, 0.6, simple_model)
            assert result is False
    
    def test_early_stopping_max_epochs(self, simple_model) -> Any:
        """Test early stopping with maximum epochs."""
        config = EarlyStoppingConfig(
            patience=10,
            max_epochs=5
        )
        early_stopping = EarlyStopping(config)
        
        # Should stop at max_epochs
        for epoch in range(6):
            result = early_stopping(epoch, 0.6, simple_model)
            if epoch < 5:
                assert result is False
            else:
                assert result is True
    
    def test_early_stopping_baseline(self, simple_model) -> Any:
        """Test early stopping with baseline value."""
        config = EarlyStoppingConfig(
            patience=3,
            baseline=0.3
        )
        early_stopping = EarlyStopping(config)
        
        # Should start with baseline
        assert early_stopping.best_score == 0.3
        
        # Improvement should update best score
        result = early_stopping(0, 0.2, simple_model)
        assert result is False
        assert early_stopping.best_score == 0.2
    
    def test_early_stopping_cooldown(self, simple_model) -> Any:
        """Test early stopping with cooldown period."""
        config = EarlyStoppingConfig(
            patience=3,
            cooldown=2
        )
        early_stopping = EarlyStopping(config)
        
        # Set initial best score
        early_stopping.best_score = 0.5
        
        # During cooldown, should not count towards patience
        for epoch in range(1, 4):
            result = early_stopping(epoch, 0.6, simple_model)
            assert result is False
            assert early_stopping.counter <= 1  # Should not exceed cooldown
    
    def test_early_stopping_restore_weights(self, early_stopping_config, simple_model) -> Any:
        """Test early stopping weight restoration."""
        early_stopping = EarlyStopping(early_stopping_config)
        
        # Set initial weights
        initial_weights = simple_model.state_dict().copy()
        
        # First call
        early_stopping(0, 0.5, simple_model)
        
        # Modify weights
        with torch.no_grad():
            simple_model[0].weight.fill_(0.1)
        
        # Trigger early stopping
        for epoch in range(1, 7):
            early_stopping(epoch, 0.6, simple_model)
        
        # Check if weights were restored
        if early_stopping.best_weights is not None:
            simple_model.load_state_dict(early_stopping.best_weights)
    
    def test_early_stopping_training_history(self, early_stopping_config, simple_model) -> Any:
        """Test early stopping training history."""
        early_stopping = EarlyStopping(early_stopping_config)
        
        # Add some training history
        for epoch in range(5):
            early_stopping(epoch, 0.5 - epoch * 0.1, simple_model)
        
        history = early_stopping.get_training_history()
        assert len(history) == 5
        
        for i, entry in enumerate(history):
            assert entry['epoch'] == i
            assert entry['metric'] == 0.5 - i * 0.1
            assert 'lr' in entry


class TestLRSchedulerFactory:
    """Test learning rate scheduler factory."""
    
    @pytest.fixture
    def optimizer(self) -> Any:
        """Create optimizer for testing."""
        model = nn.Linear(10, 1)
        return optim.Adam(model.parameters(), lr=1e-3)
    
    @pytest.fixture
    def dataloader(self) -> Any:
        """Create dataloader for testing."""
        data = torch.randn(100, 10)
        targets = torch.randn(100, 1)
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=32)
    
    def test_step_lr_scheduler(self, optimizer) -> Any:
        """Test step LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.STEP,
            step_size=10,
            gamma=0.5
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5
    
    def test_multi_step_lr_scheduler(self, optimizer) -> Any:
        """Test multi-step LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.MULTI_STEP,
            milestones=[10, 20, 30],
            gamma=0.5
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        assert scheduler.milestones == [10, 20, 30]
        assert scheduler.gamma == 0.5
    
    def test_exponential_lr_scheduler(self, optimizer) -> Any:
        """Test exponential LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.EXPONENTIAL,
            gamma=0.95
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.95
    
    def test_cosine_annealing_lr_scheduler(self, optimizer) -> Any:
        """Test cosine annealing LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING,
            T_max=100,
            eta_min=1e-6
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-6
    
    def test_cosine_annealing_warm_restarts_lr_scheduler(self, optimizer) -> Any:
        """Test cosine annealing warm restarts LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING_WARM_RESTARTS,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 10
        assert scheduler.T_mult == 2
        assert scheduler.eta_min == 1e-6
    
    def test_reduce_on_plateau_lr_scheduler(self, optimizer) -> Any:
        """Test reduce on plateau LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-6
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.mode == "min"
        assert scheduler.factor == 0.5
        assert scheduler.patience == 10
    
    def test_one_cycle_lr_scheduler(self, optimizer, dataloader) -> Any:
        """Test one cycle LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.ONE_CYCLE,
            max_lr=1e-2,
            epochs=50,
            steps_per_epoch=len(dataloader),
            pct_start=0.3
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config, dataloader)
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        assert scheduler.max_lrs[0] == 1e-2
    
    def test_cyclic_lr_scheduler(self, optimizer) -> Any:
        """Test cyclic LR scheduler creation."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.CYCLIC,
            base_lr=1e-6,
            max_lr=1e-3,
            step_size_up=1000,
            step_size_down=1000,
            mode_cyclic="triangular"
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
        assert scheduler.base_lrs[0] == 1e-6
        assert scheduler.max_lrs[0] == 1e-3
    
    def test_custom_lr_scheduler(self, optimizer) -> Any:
        """Test custom LR scheduler creation."""
        def custom_scheduler_fn(opt) -> Any:
            return torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.8)
        
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.CUSTOM,
            custom_scheduler_fn=custom_scheduler_fn
        )
        
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 5
        assert scheduler.gamma == 0.8
    
    def test_invalid_scheduler_type(self, optimizer) -> Any:
        """Test invalid scheduler type handling."""
        config = LRSchedulerConfig(
            scheduler_type="invalid_type"
        )
        
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            LRSchedulerFactory.create_scheduler(optimizer, config)
    
    def test_one_cycle_missing_dataloader(self, optimizer) -> Any:
        """Test one cycle scheduler without dataloader."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.ONE_CYCLE,
            max_lr=1e-2,
            epochs=50
        )
        
        with pytest.raises(ValueError, match="DataLoader required"):
            LRSchedulerFactory.create_scheduler(optimizer, config)
    
    def test_custom_scheduler_missing_function(self, optimizer) -> Any:
        """Test custom scheduler without function."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.CUSTOM
        )
        
        with pytest.raises(ValueError, match="Custom scheduler function required"):
            LRSchedulerFactory.create_scheduler(optimizer, config)


class TestGradientOptimizer:
    """Test gradient optimization functionality."""
    
    @pytest.fixture
    def simple_model(self) -> Any:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    @pytest.fixture
    def config(self) -> Any:
        """Create gradient optimizer configuration."""
        return TrainingOptimizationConfig(
            gradient_clip_norm=1.0,
            gradient_clip_value=None
        )
    
    def test_gradient_optimizer_initialization(self, config) -> Any:
        """Test gradient optimizer initialization."""
        optimizer = GradientOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.gradient_norms == []
        assert optimizer.clipped_gradients == 0
    
    def test_gradient_clipping_by_norm(self, simple_model, config) -> Any:
        """Test gradient clipping by norm."""
        optimizer = GradientOptimizer(config)
        
        # Create some gradients
        loss = simple_model(torch.randn(5, 10)).sum()
        loss.backward()
        
        # Clip gradients
        optimizer.clip_gradients(simple_model)
        
        # Check that gradients were processed
        assert len(optimizer.gradient_norms) == 1
        assert optimizer.gradient_norms[0] > 0
    
    def test_gradient_clipping_by_value(self, simple_model) -> Any:
        """Test gradient clipping by value."""
        config = TrainingOptimizationConfig(
            gradient_clip_norm=None,
            gradient_clip_value=0.1
        )
        optimizer = GradientOptimizer(config)
        
        # Create some gradients
        loss = simple_model(torch.randn(5, 10)).sum()
        loss.backward()
        
        # Clip gradients
        optimizer.clip_gradients(simple_model)
        
        # Check that gradients were processed
        assert len(optimizer.gradient_norms) == 0  # No norm tracking for value clipping
    
    def test_gradient_clipping_exceeds_norm(self, simple_model, config) -> Any:
        """Test gradient clipping when norm exceeds threshold."""
        optimizer = GradientOptimizer(config)
        
        # Create large gradients
        loss = simple_model(torch.randn(5, 10)).sum()
        loss.backward()
        
        # Manually set large gradients
        with torch.no_grad():
            for param in simple_model.parameters():
                param.grad *= 10
        
        # Clip gradients
        optimizer.clip_gradients(simple_model)
        
        # Check that clipping occurred
        assert optimizer.clipped_gradients > 0
    
    def test_gradient_stats(self, simple_model, config) -> Any:
        """Test gradient statistics calculation."""
        optimizer = GradientOptimizer(config)
        
        # Create gradients multiple times
        for _ in range(3):
            loss = simple_model(torch.randn(5, 10)).sum()
            loss.backward()
            optimizer.clip_gradients(simple_model)
        
        stats = optimizer.get_gradient_stats()
        
        assert "avg_gradient_norm" in stats
        assert "max_gradient_norm" in stats
        assert "min_gradient_norm" in stats
        assert "clipped_gradients_ratio" in stats
        
        assert stats["avg_gradient_norm"] > 0
        assert stats["max_gradient_norm"] >= stats["min_gradient_norm"]
        assert 0 <= stats["clipped_gradients_ratio"] <= 1


class TestTrainingMonitor:
    """Test training monitoring functionality."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create training monitor configuration."""
        return TrainingOptimizationConfig(
            log_interval=5,
            tensorboard_logging=False,
            wandb_logging=False
        )
    
    def test_training_monitor_initialization(self, config) -> Any:
        """Test training monitor initialization."""
        monitor = TrainingMonitor(config)
        
        assert monitor.config == config
        assert monitor.metrics_history == {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': []
        }
        assert monitor.start_time > 0
        assert monitor.writer is None
        assert monitor.wandb is None
    
    def test_log_metrics(self, config) -> Any:
        """Test metrics logging."""
        monitor = TrainingMonitor(config)
        
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.4,
            'train_accuracy': 0.8,
            'val_accuracy': 0.85
        }
        learning_rate = 1e-3
        
        monitor.log_metrics(0, metrics, learning_rate)
        
        assert monitor.metrics_history['train_loss'] == [0.5]
        assert monitor.metrics_history['val_loss'] == [0.4]
        assert monitor.metrics_history['train_accuracy'] == [0.8]
        assert monitor.metrics_history['val_accuracy'] == [0.85]
        assert monitor.metrics_history['learning_rate'] == [1e-3]
    
    def test_log_epoch_time(self, config) -> Any:
        """Test epoch time logging."""
        monitor = TrainingMonitor(config)
        
        monitor.log_epoch_time(0, 1.5)
        
        assert monitor.metrics_history['epoch_time'] == [1.5]
    
    def test_get_training_summary(self, config) -> Optional[Dict[str, Any]]:
        """Test training summary generation."""
        monitor = TrainingMonitor(config)
        
        # Add some metrics
        for epoch in range(3):
            metrics = {
                'train_loss': 0.5 - epoch * 0.1,
                'val_loss': 0.4 - epoch * 0.1,
                'train_accuracy': 0.8 + epoch * 0.05,
                'val_accuracy': 0.85 + epoch * 0.05
            }
            monitor.log_metrics(epoch, metrics, 1e-3)
            monitor.log_epoch_time(epoch, 1.0)
        
        summary = monitor.get_training_summary()
        
        assert summary['total_epochs'] == 3
        assert summary['avg_epoch_time'] == 1.0
        assert summary['final_train_loss'] == 0.3
        assert summary['final_val_loss'] == 0.2
        assert summary['best_val_loss'] == 0.2
        assert summary['final_train_accuracy'] == 0.9
        assert summary['final_val_accuracy'] == 0.95
        assert summary['best_val_accuracy'] == 0.95
    
    def test_close(self, config) -> Any:
        """Test monitor closing."""
        monitor = TrainingMonitor(config)
        
        # Should not raise any errors
        monitor.close()


class TestOptimizedTrainer:
    """Test optimized trainer functionality."""
    
    @pytest.fixture
    def simple_model(self) -> Any:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def simple_dataset(self) -> Any:
        """Create a simple dataset for testing."""
        data = torch.randn(100, 10)
        targets = torch.randint(0, 2, (100,))
        return TensorDataset(data, targets)
    
    @pytest.fixture
    def simple_dataloader(self, simple_dataset) -> Any:
        """Create a simple dataloader for testing."""
        return DataLoader(simple_dataset, batch_size=16, shuffle=True)
    
    @pytest.fixture
    def config(self) -> Any:
        """Create trainer configuration."""
        return TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(patience=5),
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                patience=3,
                factor=0.5
            ),
            gradient_clip_norm=1.0,
            save_checkpoints=True,
            checkpoint_dir="./test_checkpoints"
        )
    
    @pytest.fixture
    def temp_checkpoint_dir(self) -> Any:
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_trainer_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = OptimizedTrainer(config)
        
        assert trainer.config == config
        assert isinstance(trainer.early_stopping, EarlyStopping)
        assert isinstance(trainer.gradient_optimizer, GradientOptimizer)
        assert isinstance(trainer.monitor, TrainingMonitor)
    
    @pytest.mark.asyncio
    async def test_basic_training(self, simple_model, simple_dataloader, config, temp_checkpoint_dir) -> Any:
        """Test basic training functionality."""
        config.checkpoint_dir = temp_checkpoint_dir
        trainer = OptimizedTrainer(config)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=5, device=torch.device('cpu')
        )
        
        assert 'total_training_time' in summary
        assert 'total_epochs' in summary
        assert 'best_val_loss' in summary
        assert 'best_val_accuracy' in summary
        assert 'gradient_stats' in summary
        assert 'early_stopping_history' in summary
    
    @pytest.mark.asyncio
    async def test_early_stopping_triggered(self, simple_model, simple_dataloader, temp_checkpoint_dir) -> Any:
        """Test that early stopping is triggered."""
        config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(patience=3),
            save_checkpoints=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer = OptimizedTrainer(config)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train with early stopping
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=10, device=torch.device('cpu')
        )
        
        # Should stop early due to patience
        assert summary['total_epochs'] <= 10
    
    @pytest.mark.asyncio
    async def test_lr_scheduling(self, simple_model, simple_dataloader, temp_checkpoint_dir) -> Any:
        """Test learning rate scheduling."""
        config = TrainingOptimizationConfig(
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.STEP,
                step_size=2,
                gamma=0.5
            ),
            save_checkpoints=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer = OptimizedTrainer(config)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train with LR scheduling
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=5, device=torch.device('cpu')
        )
        
        # Check that LR was scheduled
        lr_history = trainer.monitor.metrics_history['learning_rate']
        assert len(lr_history) > 0
    
    @pytest.mark.asyncio
    async def test_gradient_clipping(self, simple_model, simple_dataloader, temp_checkpoint_dir) -> Any:
        """Test gradient clipping."""
        config = TrainingOptimizationConfig(
            gradient_clip_norm=0.1,  # Very small norm to trigger clipping
            save_checkpoints=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer = OptimizedTrainer(config)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train with gradient clipping
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=3, device=torch.device('cpu')
        )
        
        # Check gradient statistics
        gradient_stats = summary['gradient_stats']
        assert 'clipped_gradients_ratio' in gradient_stats
    
    @pytest.mark.asyncio
    async def test_checkpoint_saving(self, simple_model, simple_dataloader, temp_checkpoint_dir) -> Any:
        """Test checkpoint saving."""
        config = TrainingOptimizationConfig(
            save_checkpoints=True,
            save_best_only=True,
            save_last=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer = OptimizedTrainer(config)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train with checkpointing
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=3, device=torch.device('cpu')
        )
        
        # Check that checkpoints were created
        checkpoint_dir = Path(temp_checkpoint_dir)
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0
    
    @pytest.mark.asyncio
    async def test_training_analysis(self, simple_model, simple_dataloader, temp_checkpoint_dir) -> Any:
        """Test training analysis functionality."""
        config = TrainingOptimizationConfig(
            save_checkpoints=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer = OptimizedTrainer(config)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=3, device=torch.device('cpu')
        )
        
        # Test analysis plotting (should not raise errors)
        trainer.plot_training_analysis()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, simple_model, simple_dataloader, temp_checkpoint_dir) -> Any:
        """Test error handling in training."""
        config = TrainingOptimizationConfig(
            save_checkpoints=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer = OptimizedTrainer(config)
        
        # Create optimizer with invalid learning rate
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-10)
        criterion = nn.CrossEntropyLoss()
        
        # Training should handle errors gracefully
        summary = await trainer.train(
            simple_model, simple_dataloader, simple_dataloader,
            optimizer, criterion, num_epochs=2, device=torch.device('cpu')
        )
        
        assert 'total_training_time' in summary


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_optimized_trainer(self) -> Any:
        """Test create_optimized_trainer function."""
        trainer = create_optimized_trainer(
            early_stopping_patience=10,
            lr_scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
            initial_lr=1e-3,
            gradient_clip_norm=1.0
        )
        
        assert isinstance(trainer, OptimizedTrainer)
        assert trainer.config.early_stopping.patience == 10
        assert trainer.config.lr_scheduler.scheduler_type == LRSchedulerType.REDUCE_ON_PLATEAU
        assert trainer.config.gradient_clip_norm == 1.0
    
    def test_load_checkpoint(self, temp_checkpoint_dir) -> Any:
        """Test load_checkpoint function."""
        # Create a simple model and optimizer
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Create a dummy checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': {'val_loss': 0.5, 'val_accuracy': 0.8},
            'config': {}
        }
        
        checkpoint_path = Path(temp_checkpoint_dir) / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = load_checkpoint(model, optimizer, str(checkpoint_path))
        
        assert loaded_checkpoint['epoch'] == 5
        assert loaded_checkpoint['metrics']['val_loss'] == 0.5
        assert loaded_checkpoint['metrics']['val_accuracy'] == 0.8


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def integration_dataset(self) -> Any:
        """Create dataset for integration testing."""
        # Create a larger, more realistic dataset
        data = torch.randn(500, 20)
        targets = torch.randint(0, 3, (500,))
        return TensorDataset(data, targets)
    
    @pytest.fixture
    def integration_dataloader(self, integration_dataset) -> Any:
        """Create dataloader for integration testing."""
        return DataLoader(integration_dataset, batch_size=32, shuffle=True)
    
    @pytest.mark.asyncio
    async def test_complete_training_pipeline(self, integration_dataloader, temp_checkpoint_dir) -> Any:
        """Test complete training pipeline with all features."""
        # Create comprehensive configuration
        config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(
                patience=10,
                min_delta=1e-4,
                restore_best_weights=True
            ),
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                patience=5,
                factor=0.5,
                min_lr=1e-6
            ),
            gradient_clip_norm=1.0,
            gradient_accumulation_steps=2,
            save_checkpoints=True,
            save_best_only=True,
            save_last=True,
            checkpoint_dir=temp_checkpoint_dir,
            validation_frequency=1,
            log_interval=2
        )
        
        # Create trainer
        trainer = OptimizedTrainer(config)
        
        # Create model
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        summary = await trainer.train(
            model, integration_dataloader, integration_dataloader,
            optimizer, criterion, num_epochs=20, device=torch.device('cpu')
        )
        
        # Verify all components worked
        assert summary['total_epochs'] > 0
        assert summary['total_training_time'] > 0
        assert summary['best_val_loss'] is not None
        assert summary['best_val_accuracy'] is not None
        assert 'gradient_stats' in summary
        assert 'early_stopping_history' in summary
        
        # Check that checkpoints were created
        checkpoint_dir = Path(temp_checkpoint_dir)
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0
        
        # Check training history
        assert len(trainer.monitor.metrics_history['train_loss']) > 0
        assert len(trainer.monitor.metrics_history['val_loss']) > 0
        assert len(trainer.monitor.metrics_history['learning_rate']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self, integration_dataloader, temp_checkpoint_dir) -> Any:
        """Test performance comparison between different configurations."""
        configurations = [
            ("Baseline", TrainingOptimizationConfig()),
            ("Optimized", TrainingOptimizationConfig(
                early_stopping=EarlyStoppingConfig(patience=8),
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                    patience=4,
                    factor=0.5
                ),
                gradient_clip_norm=1.0
            )),
            ("Highly Optimized", TrainingOptimizationConfig(
                early_stopping=EarlyStoppingConfig(patience=12, min_delta=1e-4),
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.COSINE_ANNEALING,
                    T_max=20,
                    eta_min=1e-6
                ),
                gradient_clip_norm=1.0,
                gradient_accumulation_steps=2
            ))
        ]
        
        results = {}
        
        for name, config in configurations:
            config.checkpoint_dir = str(Path(temp_checkpoint_dir) / name.replace(" ", "_"))
            config.save_checkpoints = True
            
            trainer = OptimizedTrainer(config)
            
            model = nn.Sequential(
                nn.Linear(20, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            summary = await trainer.train(
                model, integration_dataloader, integration_dataloader,
                optimizer, criterion, num_epochs=15, device=torch.device('cpu')
            )
            
            results[name] = {
                "total_epochs": summary["total_epochs"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "total_training_time": summary["total_training_time"]
            }
        
        # Verify that all configurations completed training
        for name, result in results.items():
            assert result["total_epochs"] > 0
            assert result["best_val_accuracy"] is not None
            assert result["total_training_time"] > 0
        
        # Find best performing configuration
        best_config = max(results.items(), key=lambda x: x[1]["best_val_accuracy"])
        assert best_config[1]["best_val_accuracy"] > 0


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 
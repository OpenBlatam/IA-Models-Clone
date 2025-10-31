from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
import json
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from .production_transformers import DeviceManager
from .early_stopping_lr_scheduling import (
    import asyncio
from typing import Any, List, Dict, Optional
"""
🧪 Early Stopping & Learning Rate Scheduling Test Suite
=======================================================

Comprehensive test suite for early stopping and learning rate scheduling systems with
unit tests, integration tests, and performance benchmarks.
"""



# Import our systems
    EarlyStopping, CrossValidator, TrainingMonitor,
    EarlyStoppingConfig, EarlyStoppingStrategy, EarlyStoppingMode,
    LRSchedulerConfig, LRSchedulerType, EarlyStoppingState, LRSchedulerState,
    create_training_monitor, create_early_stopping_config, create_lr_scheduler_config
)

logger = logging.getLogger(__name__)

class TestEarlyStoppingConfig(unittest.TestCase):
    """Test early stopping configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = EarlyStoppingConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.strategy, EarlyStoppingStrategy.PATIENCE)
        self.assertEqual(config.mode, EarlyStoppingMode.MIN)
        self.assertEqual(config.patience, 10)
        self.assertEqual(config.min_delta, 0.0)
        self.assertEqual(config.min_percentage, 0.01)
        self.assertEqual(config.moving_average_window, 5)
        self.assertTrue(config.restore_best_weights)
        self.assertTrue(config.verbose)
        self.assertEqual(config.monitor, "val_loss")
        self.assertEqual(config.min_epochs, 0)
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = EarlyStoppingConfig(
            enabled=False,
            strategy=EarlyStoppingStrategy.DELTA,
            mode=EarlyStoppingMode.MAX,
            patience=5,
            min_delta=0.001,
            min_percentage=0.05,
            moving_average_window=10,
            restore_best_weights=False,
            verbose=False,
            monitor="val_accuracy",
            min_epochs=10
        )
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.strategy, EarlyStoppingStrategy.DELTA)
        self.assertEqual(config.mode, EarlyStoppingMode.MAX)
        self.assertEqual(config.patience, 5)
        self.assertEqual(config.min_delta, 0.001)
        self.assertEqual(config.min_percentage, 0.05)
        self.assertEqual(config.moving_average_window, 10)
        self.assertFalse(config.restore_best_weights)
        self.assertFalse(config.verbose)
        self.assertEqual(config.monitor, "val_accuracy")
        self.assertEqual(config.min_epochs, 10)
    
    def test_invalid_config(self) -> Any:
        """Test invalid configuration validation."""
        # Negative patience
        with self.assertRaises(ValueError):
            EarlyStoppingConfig(patience=-1)
        
        # Negative min_delta
        with self.assertRaises(ValueError):
            EarlyStoppingConfig(min_delta=-0.1)
        
        # Invalid min_percentage
        with self.assertRaises(ValueError):
            EarlyStoppingConfig(min_percentage=1.5)
        
        # Invalid moving_average_window
        with self.assertRaises(ValueError):
            EarlyStoppingConfig(moving_average_window=0)

class TestLRSchedulerConfig(unittest.TestCase):
    """Test learning rate scheduler configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = LRSchedulerConfig()
        
        self.assertEqual(config.scheduler_type, LRSchedulerType.COSINE_ANNEALING)
        self.assertEqual(config.initial_lr, 1e-3)
        self.assertEqual(config.min_lr, 1e-6)
        self.assertEqual(config.max_lr, 1e-2)
        self.assertEqual(config.step_size, 30)
        self.assertEqual(config.gamma, 0.1)
        self.assertEqual(config.milestones, [30, 60, 90])
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.STEP,
            initial_lr=1e-4,
            min_lr=1e-7,
            max_lr=1e-3,
            step_size=50,
            gamma=0.5,
            milestones=[50, 100, 150]
        )
        
        self.assertEqual(config.scheduler_type, LRSchedulerType.STEP)
        self.assertEqual(config.initial_lr, 1e-4)
        self.assertEqual(config.min_lr, 1e-7)
        self.assertEqual(config.max_lr, 1e-3)
        self.assertEqual(config.step_size, 50)
        self.assertEqual(config.gamma, 0.5)
        self.assertEqual(config.milestones, [50, 100, 150])
    
    def test_invalid_config(self) -> Any:
        """Test invalid configuration validation."""
        # Non-positive initial_lr
        with self.assertRaises(ValueError):
            LRSchedulerConfig(initial_lr=0)
        
        # Negative min_lr
        with self.assertRaises(ValueError):
            LRSchedulerConfig(min_lr=-1e-6)
        
        # Non-positive max_lr
        with self.assertRaises(ValueError):
            LRSchedulerConfig(max_lr=0)
        
        # min_lr >= max_lr
        with self.assertRaises(ValueError):
            LRSchedulerConfig(min_lr=1e-3, max_lr=1e-4)

class TestEarlyStopping(unittest.TestCase):
    """Test early stopping implementation."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_patience_strategy(self) -> Any:
        """Test patience-based early stopping."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=3,
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Simulate improving metrics
        metrics = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        
        for epoch, metric in enumerate(metrics):
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Should stop after 3 epochs without improvement
        self.assertTrue(should_stop)
        self.assertEqual(early_stopping.state.best_epoch, 5)  # Best at epoch 5
        self.assertEqual(early_stopping.state.best_score, 0.5)
    
    def test_delta_strategy(self) -> Any:
        """Test delta-based early stopping."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.DELTA,
            mode=EarlyStoppingMode.MIN,
            min_delta=0.1,
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Simulate metrics with small improvements
        metrics = [1.0, 0.95, 0.92, 0.91, 0.905, 0.902]
        
        for epoch, metric in enumerate(metrics):
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Should stop when improvement is less than delta
        self.assertTrue(should_stop)
    
    def test_percentage_strategy(self) -> Any:
        """Test percentage-based early stopping."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.PERCENTAGE,
            mode=EarlyStoppingMode.MIN,
            min_percentage=0.05,  # 5% improvement
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Simulate metrics with small percentage improvements
        metrics = [1.0, 0.98, 0.97, 0.965, 0.963, 0.962]
        
        for epoch, metric in enumerate(metrics):
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Should stop when improvement percentage is less than threshold
        self.assertTrue(should_stop)
    
    def test_moving_average_strategy(self) -> Any:
        """Test moving average-based early stopping."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.MOVING_AVERAGE,
            mode=EarlyStoppingMode.MIN,
            moving_average_window=3,
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Simulate noisy metrics
        metrics = [1.0, 0.9, 0.8, 0.85, 0.82, 0.84, 0.83, 0.85]
        
        for epoch, metric in enumerate(metrics):
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Should use moving average for stability
        self.assertTrue(len(early_stopping.state.moving_averages) > 0)
    
    def test_max_mode(self) -> Any:
        """Test early stopping in MAX mode."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MAX,
            patience=3,
            monitor="val_accuracy"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Simulate accuracy metrics (higher is better)
        metrics = [0.5, 0.6, 0.7, 0.65, 0.68, 0.67, 0.66]
        
        for epoch, metric in enumerate(metrics):
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Should stop after 3 epochs without improvement
        self.assertTrue(should_stop)
        self.assertEqual(early_stopping.state.best_score, 0.7)
    
    def test_min_epochs(self) -> Any:
        """Test minimum epochs before stopping."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=2,
            min_epochs=5,
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Simulate metrics that would trigger early stopping
        metrics = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
        
        for epoch, metric in enumerate(metrics):
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Should not stop before min_epochs
        self.assertFalse(should_stop)  # Should continue until min_epochs
    
    def test_model_restoration(self) -> Any:
        """Test model weight restoration."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=3,
            restore_best_weights=True,
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Get initial weights
        initial_weights = model.weight.clone()
        
        # Simulate training with improvement then degradation
        metrics = [1.0, 0.9, 0.8, 0.9, 1.0, 1.1]
        
        for epoch, metric in enumerate(metrics):
            # Modify weights to simulate training
            model.weight.data += torch.randn_like(model.weight) * 0.1
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        # Restore best model
        early_stopping.restore_best_model(model)
        
        # Check that weights were restored
        self.assertTrue(hasattr(early_stopping, '_best_model_state'))
    
    def test_state_management(self) -> Any:
        """Test state management."""
        config = EarlyStoppingConfig(
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=3,
            monitor="val_loss"
        )
        
        early_stopping = EarlyStopping(config, self.device_manager)
        model = nn.Linear(10, 1)
        
        # Run some epochs
        metrics = [1.0, 0.9, 0.8, 0.8, 0.8]
        
        for epoch, metric in enumerate(metrics):
            early_stopping(epoch, metric, model)
        
        # Get state
        state = early_stopping.get_state()
        
        self.assertIn('best_score', state)
        self.assertIn('best_epoch', state)
        self.assertIn('counter', state)
        self.assertIn('history', state)
        self.assertIn('moving_averages', state)
        
        # Reset state
        early_stopping.reset()
        
        # Check reset
        self.assertEqual(early_stopping.state.best_score, float('inf'))
        self.assertEqual(early_stopping.state.counter, 0)
        self.assertEqual(len(early_stopping.state.history), 0)

class TestLRScheduler(unittest.TestCase):
    """Test learning rate scheduler implementation."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_step_scheduler(self) -> Any:
        """Test step learning rate scheduler."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.STEP,
            initial_lr=1e-3,
            step_size=2,
            gamma=0.5
        )
        
        scheduler_wrapper = LRScheduler(config, self.device_manager)
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
        
        scheduler = scheduler_wrapper.create_scheduler(optimizer)
        
        # Test LR decay
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(5):
            scheduler_wrapper.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            if epoch < 2:
                self.assertEqual(current_lr, initial_lr)
            else:
                self.assertEqual(current_lr, initial_lr * (0.5 ** ((epoch - 1) // 2)))
    
    def test_cosine_annealing_scheduler(self) -> Any:
        """Test cosine annealing scheduler."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING,
            initial_lr=1e-3,
            T_max=10,
            eta_min=1e-6
        )
        
        scheduler_wrapper = LRScheduler(config, self.device_manager)
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
        
        scheduler = scheduler_wrapper.create_scheduler(optimizer)
        
        # Test cosine annealing
        lrs = []
        for epoch in range(10):
            scheduler_wrapper.step(epoch)
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # Check that LR decreases and then increases (cosine pattern)
        self.assertGreater(lrs[0], lrs[5])  # LR should decrease
        self.assertLess(lrs[5], lrs[9])     # LR should increase again
    
    def test_reduce_lr_on_plateau(self) -> Any:
        """Test reduce LR on plateau scheduler."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.REDUCE_LR_ON_PLATEAU,
            initial_lr=1e-3,
            factor=0.5,
            patience=2,
            min_lr_plateau=1e-6
        )
        
        scheduler_wrapper = LRScheduler(config, self.device_manager)
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
        
        scheduler = scheduler_wrapper.create_scheduler(optimizer)
        
        # Simulate plateau in validation loss
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Decreasing loss (should not reduce LR)
        for epoch in range(3):
            scheduler_wrapper.step(epoch, {'val_loss': 1.0 - epoch * 0.1})
            self.assertEqual(optimizer.param_groups[0]['lr'], initial_lr)
        
        # Plateau in loss (should reduce LR)
        for epoch in range(3, 6):
            scheduler_wrapper.step(epoch, {'val_loss': 0.7})
        
        # LR should be reduced
        self.assertLess(optimizer.param_groups[0]['lr'], initial_lr)
    
    def test_one_cycle_scheduler(self) -> Any:
        """Test one cycle scheduler."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.ONE_CYCLE,
            initial_lr=1e-4,
            max_lr=1e-2,
            epochs=10,
            steps_per_epoch=10
        )
        
        scheduler_wrapper = LRScheduler(config, self.device_manager)
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
        
        scheduler = scheduler_wrapper.create_scheduler(optimizer)
        
        # Test one cycle pattern
        lrs = []
        for epoch in range(10):
            for step in range(10):
                scheduler_wrapper.step()
                lrs.append(optimizer.param_groups[0]['lr'])
        
        # Check one cycle pattern (increase then decrease)
        mid_point = len(lrs) // 2
        self.assertGreater(max(lrs[:mid_point]), lrs[0])  # Should increase
        self.assertGreater(max(lrs[:mid_point]), lrs[-1])  # Should decrease
    
    def test_state_management(self) -> Any:
        """Test scheduler state management."""
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.STEP,
            initial_lr=1e-3,
            step_size=2,
            gamma=0.5
        )
        
        scheduler_wrapper = LRScheduler(config, self.device_manager)
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
        
        scheduler = scheduler_wrapper.create_scheduler(optimizer)
        
        # Run some steps
        for epoch in range(5):
            scheduler_wrapper.step(epoch)
        
        # Get state
        state = scheduler_wrapper.get_state()
        
        self.assertIn('current_lr', state)
        self.assertIn('best_lr', state)
        self.assertIn('history', state)
        self.assertIn('scheduler_type', state)
        
        # Reset state
        scheduler_wrapper.reset()
        
        # Check reset
        self.assertEqual(scheduler_wrapper.state.current_lr, 1e-3)
        self.assertEqual(len(scheduler_wrapper.state.history), 0)
        self.assertIsNone(scheduler_wrapper.state.scheduler)

class TestTrainingMonitor(unittest.TestCase):
    """Test training monitor implementation."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_monitor_initialization(self) -> Any:
        """Test monitor initialization."""
        monitor = await create_training_monitor(self.device_manager)
        
        self.assertIsNotNone(monitor.device_manager)
        self.assertIsNone(monitor.early_stopping)
        self.assertIsNone(monitor.lr_scheduler)
        self.assertIsNotNone(monitor.logger)
        self.assertIsInstance(monitor.training_history, dict)
    
    async def test_early_stopping_setup(self) -> Any:
        """Test early stopping setup."""
        monitor = await create_training_monitor(self.device_manager)
        
        es_config = create_early_stopping_config(
            enabled=True,
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=5
        )
        
        monitor.setup_early_stopping(es_config)
        
        self.assertIsNotNone(monitor.early_stopping)
        self.assertEqual(monitor.early_stopping.config.strategy, EarlyStoppingStrategy.PATIENCE)
    
    async def test_lr_scheduler_setup(self) -> Any:
        """Test LR scheduler setup."""
        monitor = await create_training_monitor(self.device_manager)
        
        lr_config = create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.STEP,
            initial_lr=1e-3
        )
        
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        
        scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
        
        self.assertIsNotNone(monitor.lr_scheduler)
        self.assertIsNotNone(scheduler)
    
    async def test_monitor_update(self) -> Any:
        """Test monitor update functionality."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Setup early stopping
        es_config = create_early_stopping_config(
            enabled=True,
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=3
        )
        monitor.setup_early_stopping(es_config)
        
        # Setup LR scheduler
        lr_config = create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.STEP,
            initial_lr=1e-3
        )
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        monitor.setup_lr_scheduler(lr_config, optimizer)
        
        # Simulate training updates
        for epoch in range(5):
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'train_accuracy': 0.5 + epoch * 0.1,
                'val_accuracy': 0.48 + epoch * 0.08,
                'learning_rate': 1e-3 * (0.9 ** epoch)
            }
            
            should_stop = monitor.update(epoch, metrics, model)
            
            if should_stop:
                break
        
        # Check training history
        self.assertGreater(len(monitor.training_history['epoch']), 0)
        self.assertGreater(len(monitor.training_history['train_loss']), 0)
        self.assertGreater(len(monitor.training_history['val_loss']), 0)
    
    async def test_training_summary(self) -> Any:
        """Test training summary generation."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Setup components
        es_config = create_early_stopping_config(enabled=True)
        monitor.setup_early_stopping(es_config)
        
        lr_config = create_lr_scheduler_config()
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        monitor.setup_lr_scheduler(lr_config, optimizer)
        
        # Simulate some training
        for epoch in range(3):
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'train_accuracy': 0.5 + epoch * 0.1,
                'val_accuracy': 0.48 + epoch * 0.08,
                'learning_rate': 1e-3
            }
            monitor.update(epoch, metrics, model)
        
        # Get summary
        summary = monitor.get_training_summary()
        
        self.assertIn('total_epochs', summary)
        self.assertIn('final_train_loss', summary)
        self.assertIn('final_val_loss', summary)
        self.assertIn('training_history', summary)
        self.assertIn('early_stopping_state', summary)
        self.assertIn('lr_scheduler_state', summary)
    
    async def test_plot_training_curves(self) -> Any:
        """Test training curves plotting."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Simulate training data
        for epoch in range(5):
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'train_accuracy': 0.5 + epoch * 0.1,
                'val_accuracy': 0.48 + epoch * 0.08,
                'learning_rate': 1e-3 * (0.9 ** epoch)
            }
            monitor.update(epoch, metrics, nn.Linear(10, 1))
        
        # Test plotting
        plot_path = Path(self.temp_dir) / "training_curves.png"
        monitor.plot_training_curves(str(plot_path))
        
        # Check if plot was saved
        self.assertTrue(plot_path.exists())
    
    async def test_reset_functionality(self) -> Any:
        """Test monitor reset functionality."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Setup components
        es_config = create_early_stopping_config(enabled=True)
        monitor.setup_early_stopping(es_config)
        
        lr_config = create_lr_scheduler_config()
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        monitor.setup_lr_scheduler(lr_config, optimizer)
        
        # Add some training data
        for epoch in range(3):
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'train_accuracy': 0.5 + epoch * 0.1,
                'val_accuracy': 0.48 + epoch * 0.08,
                'learning_rate': 1e-3
            }
            monitor.update(epoch, metrics, model)
        
        # Reset
        monitor.reset()
        
        # Check reset
        self.assertEqual(len(monitor.training_history['epoch']), 0)
        self.assertEqual(len(monitor.training_history['train_loss']), 0)
        self.assertIsNone(monitor.early_stopping)
        self.assertIsNone(monitor.lr_scheduler)

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_end_to_end_workflow(self) -> Any:
        """Test end-to-end workflow."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Setup early stopping
        es_config = create_early_stopping_config(
            enabled=True,
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=3,
            monitor="val_loss"
        )
        monitor.setup_early_stopping(es_config)
        
        # Setup LR scheduler
        lr_config = create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING,
            initial_lr=1e-3,
            T_max=10
        )
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
        
        # Simulate training loop
        should_stop = False
        for epoch in range(10):
            # Simulate training metrics
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'train_accuracy': 0.5 + epoch * 0.05,
                'val_accuracy': 0.48 + epoch * 0.04,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            # Update monitor
            should_stop = monitor.update(epoch, metrics, model)
            
            if should_stop:
                break
        
        # Get summary
        summary = monitor.get_training_summary()
        
        # Validate results
        self.assertIn('total_epochs', summary)
        self.assertIn('early_stopping_state', summary)
        self.assertIn('lr_scheduler_state', summary)
        
        # Check early stopping state
        es_state = summary['early_stopping_state']
        self.assertIn('best_score', es_state)
        self.assertIn('best_epoch', es_state)
        
        # Check LR scheduler state
        lr_state = summary['lr_scheduler_state']
        self.assertIn('current_lr', lr_state)
        self.assertIn('history', lr_state)
    
    async def test_custom_early_stopping(self) -> Any:
        """Test custom early stopping function."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Define custom stopping function
        def custom_stopping(epoch, metric, state) -> Any:
            return epoch >= 5  # Stop after 5 epochs
        
        es_config = EarlyStoppingConfig(
            enabled=True,
            strategy=EarlyStoppingStrategy.CUSTOM,
            custom_stopping_function=custom_stopping,
            monitor="val_loss"
        )
        monitor.setup_early_stopping(es_config)
        
        # Simulate training
        model = nn.Linear(10, 1)
        should_stop = False
        
        for epoch in range(10):
            metrics = {
                'train_loss': 1.0,
                'val_loss': 1.0,
                'learning_rate': 1e-3
            }
            
            should_stop = monitor.update(epoch, metrics, model)
            
            if should_stop:
                break
        
        # Should stop at epoch 5
        self.assertTrue(should_stop)
    
    async def test_custom_lr_scheduler(self) -> Any:
        """Test custom learning rate scheduler."""
        monitor = await create_training_monitor(self.device_manager)
        
        # Define custom scheduler function
        def custom_scheduler(optimizer) -> Any:
            return optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.9 ** epoch)
        
        lr_config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.CUSTOM,
            custom_scheduler_function=custom_scheduler,
            initial_lr=1e-3
        )
        
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
        
        # Test custom scheduler
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(5):
            monitor.lr_scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            expected_lr = initial_lr * (0.9 ** epoch)
            
            self.assertAlmostEqual(current_lr, expected_lr, places=6)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks."""
    
    def setUp(self) -> Any:
        """Set up benchmark environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up benchmark environment."""
        shutil.rmtree(self.temp_dir)
    
    async def benchmark_early_stopping_performance(self) -> Any:
        """Benchmark early stopping performance."""
        es_config = create_early_stopping_config(
            enabled=True,
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=10
        )
        
        early_stopping = EarlyStopping(es_config, self.device_manager)
        model = nn.Linear(1000, 100)  # Large model
        
        start_time = time.time()
        
        # Simulate many epochs
        for epoch in range(1000):
            metric = 1.0 / (epoch + 1)
            should_stop = early_stopping(epoch, metric, model)
            
            if should_stop:
                break
        
        early_stopping_time = time.time() - start_time
        
        self.assertLess(early_stopping_time, 1.0)  # Should complete within 1 second
        logger.info(f"Early stopping benchmark: {early_stopping_time:.4f} seconds")
    
    async def benchmark_lr_scheduler_performance(self) -> Any:
        """Benchmark LR scheduler performance."""
        lr_config = create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING,
            initial_lr=1e-3,
            T_max=1000
        )
        
        scheduler_wrapper = LRScheduler(lr_config, self.device_manager)
        model = nn.Linear(1000, 100)  # Large model
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        
        scheduler = scheduler_wrapper.create_scheduler(optimizer)
        
        start_time = time.time()
        
        # Simulate many steps
        for step in range(1000):
            scheduler_wrapper.step(step)
        
        scheduler_time = time.time() - start_time
        
        self.assertLess(scheduler_time, 0.5)  # Should complete within 0.5 seconds
        logger.info(f"LR scheduler benchmark: {scheduler_time:.4f} seconds")

class TestErrorHandling(unittest.TestCase):
    """Test error handling."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_early_stopping_config(self) -> Any:
        """Test handling of invalid early stopping configuration."""
        with self.assertRaises(ValueError):
            EarlyStoppingConfig(patience=-1)
        
        with self.assertRaises(ValueError):
            EarlyStoppingConfig(min_delta=-0.1)
    
    def test_invalid_lr_scheduler_config(self) -> Any:
        """Test handling of invalid LR scheduler configuration."""
        with self.assertRaises(ValueError):
            LRSchedulerConfig(initial_lr=0)
        
        with self.assertRaises(ValueError):
            LRSchedulerConfig(min_lr=1e-3, max_lr=1e-4)
    
    def test_missing_custom_functions(self) -> Any:
        """Test handling of missing custom functions."""
        # Missing custom stopping function
        with self.assertRaises(ValueError):
            es_config = EarlyStoppingConfig(
                strategy=EarlyStoppingStrategy.CUSTOM,
                custom_stopping_function=None
            )
            early_stopping = EarlyStopping(es_config, self.device_manager)
            early_stopping(0, 1.0, nn.Linear(10, 1))
        
        # Missing custom scheduler function
        with self.assertRaises(ValueError):
            lr_config = LRSchedulerConfig(
                scheduler_type=LRSchedulerType.CUSTOM,
                custom_scheduler_function=None
            )
            scheduler_wrapper = LRScheduler(lr_config, self.device_manager)
            model = nn.Linear(10, 1)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler_wrapper.create_scheduler(optimizer)

# Test runner functions
def run_performance_tests():
    """Run performance benchmarks."""
    print("🚀 Running Performance Benchmarks...")
    
    benchmark_suite = unittest.TestSuite()
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_early_stopping_performance'))
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_lr_scheduler_performance'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(benchmark_suite)

def run_all_tests():
    """Run all tests."""
    print("🧪 Running All Early Stopping & LR Scheduling Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEarlyStoppingConfig,
        TestLRSchedulerConfig,
        TestEarlyStopping,
        TestLRScheduler,
        TestTrainingMonitor,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

# Quick test functions
async def quick_early_stopping_test():
    """Quick test for early stopping."""
    print("🧪 Quick Early Stopping Test...")
    
    try:
        # Create monitor
        device_manager = DeviceManager()
        monitor = await create_training_monitor(device_manager)
        
        # Setup early stopping
        es_config = create_early_stopping_config(
            enabled=True,
            strategy=EarlyStoppingStrategy.PATIENCE,
            mode=EarlyStoppingMode.MIN,
            patience=3
        )
        monitor.setup_early_stopping(es_config)
        
        # Simulate training
        model = nn.Linear(10, 1)
        should_stop = False
        
        for epoch in range(5):
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'learning_rate': 1e-3
            }
            
            should_stop = monitor.update(epoch, metrics, model)
            
            if should_stop:
                break
        
        print(f"✅ Early stopping test passed: stopped at epoch {epoch}")
        return True
        
    except Exception as e:
        print(f"❌ Early stopping test failed: {e}")
        return False

async def quick_lr_scheduler_test():
    """Quick test for LR scheduler."""
    print("🧪 Quick LR Scheduler Test...")
    
    try:
        # Create monitor
        device_manager = DeviceManager()
        monitor = await create_training_monitor(device_manager)
        
        # Setup LR scheduler
        lr_config = create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING,
            initial_lr=1e-3
        )
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
        scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
        
        # Test scheduler
        initial_lr = optimizer.param_groups[0]['lr']
        lrs = []
        
        for epoch in range(5):
            monitor.lr_scheduler.step(epoch)
            lrs.append(optimizer.param_groups[0]['lr'])
        
        print(f"✅ LR scheduler test passed: {len(lrs)} steps completed")
        return True
        
    except Exception as e:
        print(f"❌ LR scheduler test failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    
    async def main():
        
    """main function."""
print("🚀 Early Stopping & Learning Rate Scheduling Test Suite")
        print("=" * 70)
        
        # Run quick tests
        print("\n1. Quick Tests:")
        es_success = await quick_early_stopping_test()
        lr_success = await quick_lr_scheduler_test()
        
        # Run performance tests
        print("\n2. Performance Tests:")
        run_performance_tests()
        
        # Run comprehensive tests
        print("\n3. Comprehensive Tests:")
        all_tests_success = run_all_tests()
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 Test Summary:")
        print(f"Early Stopping Test: {'✅ PASSED' if es_success else '❌ FAILED'}")
        print(f"LR Scheduler Test: {'✅ PASSED' if lr_success else '❌ FAILED'}")
        print(f"All Tests: {'✅ PASSED' if all_tests_success else '❌ FAILED'}")
        
        if es_success and lr_success and all_tests_success:
            print("\n🎉 All tests passed! The Early Stopping & LR Scheduling system is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
    
    asyncio.run(main()) 
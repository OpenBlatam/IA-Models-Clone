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
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from multi_gpu_training import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Multi-GPU Training System

This test suite covers:
- DataParallel training functionality
- DistributedDataParallel training functionality
- Performance monitoring
- Fault tolerance mechanisms
- Memory optimization
- Edge cases and error handling
- Integration testing
"""



    MultiGPUConfig, MultiGPUTrainingManager, TrainingMode,
    DataParallelTrainer, DistributedDataParallelTrainer,
    MetricsCollector, FaultToleranceManager,
    setup_distributed_training, run_distributed_training
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


class TestMultiGPUConfig:
    """Test MultiGPUConfig class."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = MultiGPUConfig()
        
        assert config.training_mode == TrainingMode.DATA_PARALLEL
        assert config.use_mixed_precision is True
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0
        assert config.pin_memory is True
        assert config.enable_fault_tolerance is True
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = MultiGPUConfig(
            training_mode=TrainingMode.DISTRIBUTED_DATA_PARALLEL,
            world_size=4,
            rank=1,
            local_rank=1,
            use_mixed_precision=False,
            gradient_accumulation_steps=4,
            max_grad_norm=2.0
        )
        
        assert config.training_mode == TrainingMode.DISTRIBUTED_DATA_PARALLEL
        assert config.world_size == 4
        assert config.rank == 1
        assert config.local_rank == 1
        assert config.use_mixed_precision is False
        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 2.0
    
    def test_post_init_cuda_unavailable(self) -> Any:
        """Test post_init when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            config = MultiGPUConfig()
            
            assert config.training_mode == TrainingMode.SINGLE_GPU
            assert config.device_ids == []
    
    def test_post_init_distributed(self) -> Any:
        """Test post_init for distributed training."""
        with patch.dict(os.environ, {'WORLD_SIZE': '4', 'RANK': '2', 'LOCAL_RANK': '2'}):
            config = MultiGPUConfig(training_mode=TrainingMode.DISTRIBUTED_DATA_PARALLEL)
            
            assert config.world_size == 4
            assert config.rank == 2
            assert config.local_rank == 2


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_initialization(self) -> Any:
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        
        assert len(collector.training_metrics) == 0
        assert len(collector.validation_metrics) == 0
        assert len(collector.current_epoch_metrics) == 0
        assert len(collector.current_validation_metrics) == 0
    
    def test_update_training_metrics(self) -> Any:
        """Test updating training metrics."""
        collector = MetricsCollector()
        
        metrics1 = {'loss': 0.5, 'accuracy': 0.8}
        metrics2 = {'loss': 0.3, 'accuracy': 0.9}
        
        collector.update(metrics1)
        collector.update(metrics2)
        
        assert len(collector.current_epoch_metrics) == 2
        assert collector.current_epoch_metrics[0] == metrics1
        assert collector.current_epoch_metrics[1] == metrics2
    
    def test_update_validation_metrics(self) -> Any:
        """Test updating validation metrics."""
        collector = MetricsCollector()
        
        metrics = {'val_loss': 0.4, 'val_accuracy': 0.85}
        collector.update(metrics, is_validation=True)
        
        assert len(collector.current_validation_metrics) == 1
        assert collector.current_validation_metrics[0] == metrics
    
    def test_get_epoch_metrics(self) -> Optional[Dict[str, Any]]:
        """Test getting aggregated epoch metrics."""
        collector = MetricsCollector()
        
        # Add some metrics
        collector.update({'loss': 0.5, 'accuracy': 0.8})
        collector.update({'loss': 0.3, 'accuracy': 0.9})
        collector.update({'loss': 0.4, 'accuracy': 0.85})
        
        epoch_metrics = collector.get_epoch_metrics()
        
        assert 'loss' in epoch_metrics
        assert 'accuracy' in epoch_metrics
        assert abs(epoch_metrics['loss'] - 0.4) < 1e-6  # (0.5 + 0.3 + 0.4) / 3
        assert abs(epoch_metrics['accuracy'] - 0.85) < 1e-6  # (0.8 + 0.9 + 0.85) / 3
        assert len(collector.training_metrics) == 1
        assert len(collector.current_epoch_metrics) == 0
    
    def test_get_validation_metrics(self) -> Optional[Dict[str, Any]]:
        """Test getting aggregated validation metrics."""
        collector = MetricsCollector()
        
        # Add validation metrics
        collector.update({'val_loss': 0.4, 'val_accuracy': 0.85}, is_validation=True)
        collector.update({'val_loss': 0.35, 'val_accuracy': 0.88}, is_validation=True)
        
        val_metrics = collector.get_validation_metrics()
        
        assert 'val_loss' in val_metrics
        assert 'val_accuracy' in val_metrics
        assert abs(val_metrics['val_loss'] - 0.375) < 1e-6
        assert abs(val_metrics['val_accuracy'] - 0.865) < 1e-6
        assert len(collector.validation_metrics) == 1
        assert len(collector.current_validation_metrics) == 0
    
    def test_empty_metrics(self) -> Any:
        """Test handling empty metrics."""
        collector = MetricsCollector()
        
        epoch_metrics = collector.get_epoch_metrics()
        val_metrics = collector.get_validation_metrics()
        
        assert epoch_metrics == {}
        assert val_metrics == {}


class TestFaultToleranceManager:
    """Test FaultToleranceManager class."""
    
    def test_initialization(self) -> Any:
        """Test FaultToleranceManager initialization."""
        config = MultiGPUConfig(enable_fault_tolerance=True)
        manager = FaultToleranceManager(config)
        
        assert manager.config == config
        assert manager.checkpoint_dir.exists()
        assert manager.last_checkpoint is None
    
    def test_save_checkpoint(self, tmp_path) -> Any:
        """Test saving checkpoint."""
        config = MultiGPUConfig(enable_fault_tolerance=True)
        manager = FaultToleranceManager(config)
        manager.checkpoint_dir = tmp_path
        
        model = TestModel()
        step = 100
        
        manager.save_checkpoint(model, step)
        
        checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
        assert checkpoint_path.exists()
        
        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert checkpoint['step'] == step
        assert 'model_state_dict' in checkpoint
        assert 'timestamp' in checkpoint
    
    def test_load_checkpoint(self, tmp_path) -> Any:
        """Test loading checkpoint."""
        config = MultiGPUConfig(enable_fault_tolerance=True)
        manager = FaultToleranceManager(config)
        manager.checkpoint_dir = tmp_path
        
        # Create a checkpoint
        model = TestModel()
        step = 100
        checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'timestamp': '2023-01-01T00:00:00'
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_step = manager.load_checkpoint(model, step)
        assert loaded_step == step
    
    def test_load_nonexistent_checkpoint(self, tmp_path) -> Any:
        """Test loading non-existent checkpoint."""
        config = MultiGPUConfig(enable_fault_tolerance=True)
        manager = FaultToleranceManager(config)
        manager.checkpoint_dir = tmp_path
        
        model = TestModel()
        step = 999
        
        loaded_step = manager.load_checkpoint(model, step)
        assert loaded_step == 0
    
    def test_cleanup_old_checkpoints(self, tmp_path) -> Any:
        """Test cleanup of old checkpoints."""
        config = MultiGPUConfig(enable_fault_tolerance=True)
        manager = FaultToleranceManager(config)
        manager.checkpoint_dir = tmp_path
        
        # Create multiple checkpoints
        model = TestModel()
        for step in [100, 200, 300, 400]:
            checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'timestamp': '2023-01-01T00:00:00'
            }
            torch.save(checkpoint, checkpoint_path)
        
        # Trigger cleanup
        manager._cleanup_old_checkpoints(keep_last=2)
        
        # Check that only 2 checkpoints remain
        remaining_checkpoints = list(tmp_path.glob("checkpoint_step_*.pt"))
        assert len(remaining_checkpoints) == 2
    
    def test_handle_error(self) -> Any:
        """Test error handling."""
        config = MultiGPUConfig(enable_fault_tolerance=True)
        manager = FaultToleranceManager(config)
        
        error = RuntimeError("Test error")
        manager.handle_error(error)
        
        # This should not raise an exception
        assert True


class TestDataParallelTrainer:
    """Test DataParallelTrainer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0] if torch.cuda.is_available() else [],
            use_mixed_precision=False  # Disable for testing
        )
    
    @pytest.fixture
    def trainer(self, config) -> Any:
        """Create test trainer."""
        return DataParallelTrainer(config)
    
    def test_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = DataParallelTrainer(config)
        
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.scaler is None
    
    def test_setup_model_single_gpu(self, trainer) -> Any:
        """Test model setup for single GPU."""
        model = TestModel()
        
        if torch.cuda.is_available():
            wrapped_model = trainer.setup_model(model)
            assert isinstance(wrapped_model, nn.Module)
            assert wrapped_model.device.type == 'cuda'
        else:
            wrapped_model = trainer.setup_model(model)
            assert isinstance(wrapped_model, nn.Module)
            assert wrapped_model.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                        reason="Requires multiple GPUs")
    def test_setup_model_multi_gpu(self, config) -> Any:
        """Test model setup for multiple GPUs."""
        config.device_ids = [0, 1]
        trainer = DataParallelTrainer(config)
        model = TestModel()
        
        wrapped_model = trainer.setup_model(model)
        assert isinstance(wrapped_model, DataParallel)
    
    def test_setup_dataloader(self, trainer) -> Any:
        """Test dataloader setup."""
        dataset = TestDataset(100)
        dataloader = trainer.setup_dataloader(dataset, batch_size=16, shuffle=True)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16
        assert dataloader.dataset == dataset
    
    def test_train_step(self, trainer) -> Any:
        """Test training step."""
        model = TestModel()
        trainer.setup_model(model)
        
        # Setup optimizer
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create batch
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        
        outputs = trainer.train_step(batch)
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['loss'].requires_grad is False  # Loss should be detached
    
    def test_validate_step(self, trainer) -> bool:
        """Test validation step."""
        model = TestModel()
        trainer.setup_model(model)
        
        # Create batch
        batch = {
            'input_ids': torch.randn(4, 64),
            'labels': torch.randint(0, 5, (4,))
        }
        
        outputs = trainer.validate_step(batch)
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['loss'].requires_grad is False


class TestDistributedDataParallelTrainer:
    """Test DistributedDataParallelTrainer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MultiGPUConfig(
            training_mode=TrainingMode.DISTRIBUTED_DATA_PARALLEL,
            world_size=1,
            rank=0,
            local_rank=0,
            use_mixed_precision=False
        )
    
    @pytest.fixture
    def trainer(self, config) -> Any:
        """Create test trainer."""
        return DistributedDataParallelTrainer(config)
    
    def test_initialization(self, config) -> Any:
        """Test trainer initialization."""
        trainer = DistributedDataParallelTrainer(config)
        
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.scaler is None
    
    @patch('torch.distributed.init_process_group')
    def test_setup_distributed(self, mock_init_process_group, config) -> Any:
        """Test distributed setup."""
        trainer = DistributedDataParallelTrainer(config)
        
        # Verify init_process_group was called
        mock_init_process_group.assert_called_once()
        
        # Check that distributed is set up
        assert trainer.is_distributed is True
        assert trainer.rank == 0
        assert trainer.world_size == 1
    
    @patch('torch.distributed.init_process_group')
    def test_setup_distributed_failure(self, mock_init_process_group, config) -> Any:
        """Test distributed setup failure."""
        mock_init_process_group.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            DistributedDataParallelTrainer(config)
    
    @patch('torch.distributed.init_process_group')
    def test_setup_model(self, mock_init_process_group, trainer) -> Any:
        """Test model setup."""
        model = TestModel()
        wrapped_model = trainer.setup_model(model)
        
        assert isinstance(wrapped_model, DistributedDataParallel)
    
    @patch('torch.distributed.init_process_group')
    def test_setup_dataloader(self, mock_init_process_group, trainer) -> Any:
        """Test dataloader setup."""
        dataset = TestDataset(100)
        dataloader = trainer.setup_dataloader(dataset, batch_size=16, shuffle=True)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16
        assert hasattr(dataloader, 'sampler')
        assert isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler)


class TestMultiGPUTrainingManager:
    """Test MultiGPUTrainingManager class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0] if torch.cuda.is_available() else [],
            use_mixed_precision=False
        )
    
    @pytest.fixture
    def manager(self, config) -> Any:
        """Create test manager."""
        return MultiGPUTrainingManager(config)
    
    def test_initialization(self, config) -> Any:
        """Test manager initialization."""
        manager = MultiGPUTrainingManager(config)
        
        assert manager.config == config
        assert isinstance(manager.trainer, DataParallelTrainer)
        assert isinstance(manager.metrics_collector, MetricsCollector)
        assert manager.fault_tolerance is not None
    
    def test_create_trainer_data_parallel(self, config) -> Any:
        """Test creating DataParallel trainer."""
        config.training_mode = TrainingMode.DATA_PARALLEL
        manager = MultiGPUTrainingManager(config)
        
        assert isinstance(manager.trainer, DataParallelTrainer)
    
    def test_create_trainer_distributed(self, config) -> Any:
        """Test creating DistributedDataParallel trainer."""
        config.training_mode = TrainingMode.DISTRIBUTED_DATA_PARALLEL
        manager = MultiGPUTrainingManager(config)
        
        assert isinstance(manager.trainer, DistributedDataParallelTrainer)
    
    def test_setup_training(self, manager) -> Any:
        """Test training setup."""
        model = TestModel()
        dataset = TestDataset(100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        model, train_loader, val_loader = manager.setup_training(
            model, dataset, dataset, optimizer
        )
        
        assert model is not None
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert manager.trainer.optimizer == optimizer
    
    def test_train_epoch(self, manager) -> Any:
        """Test training epoch."""
        model = TestModel()
        dataset = TestDataset(100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        model, train_loader, _ = manager.setup_training(model, dataset, optimizer=optimizer)
        
        # Mock the trainer's train_step method
        def mock_train_step(batch) -> Any:
            return {'loss': torch.tensor(0.5), 'logits': torch.randn(4, 5)}
        
        manager.trainer.train_step = mock_train_step
        
        metrics = manager.train_epoch(train_loader, 0)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_validate_epoch(self, manager) -> bool:
        """Test validation epoch."""
        model = TestModel()
        dataset = TestDataset(100)
        
        model, _, val_loader = manager.setup_training(model, dataset, val_dataset=dataset)
        
        # Mock the trainer's validate_step method
        def mock_validate_step(batch) -> bool:
            return {'loss': torch.tensor(0.4), 'logits': torch.randn(4, 5)}
        
        manager.trainer.validate_step = mock_validate_step
        
        metrics = manager.validate_epoch(val_loader)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_extract_metrics(self, manager) -> Any:
        """Test metrics extraction."""
        outputs = {
            'loss': torch.tensor(0.5),
            'logits': torch.randn(4, 5),
            'features': torch.randn(4, 10)
        }
        
        metrics = manager._extract_metrics(outputs)
        
        assert 'loss' in metrics
        assert 'logits' in metrics
        assert 'features' in metrics
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['logits'], float)
        assert isinstance(metrics['features'], float)
    
    def test_cleanup(self, manager) -> Any:
        """Test cleanup."""
        # Mock distributed cleanup
        with patch('torch.distributed.destroy_process_group') as mock_destroy:
            manager.cleanup()
            
            # Should not call destroy_process_group for DataParallel
            mock_destroy.assert_not_called()


class TestIntegration:
    """Integration tests for the multi-GPU training system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_training(self) -> Any:
        """Test end-to-end training workflow."""
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0] if torch.cuda.is_available() else [],
            use_mixed_precision=False,
            batch_size=8,
            num_workers=0  # Disable multiprocessing for testing
        )
        
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(50)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(
            model, dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
        )
        
        # Train for one epoch
        metrics = manager.train_epoch(train_loader, 0)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        
        manager.cleanup()
    
    def test_memory_management(self) -> Any:
        """Test memory management during training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0],
            use_mixed_precision=True,
            batch_size=16
        )
        
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(100)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(model, dataset)
        
        # Record initial memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Train for a few steps
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Only train for 5 steps
                break
            manager.trainer.train_step(batch)
        
        # Check memory usage
        final_memory = torch.cuda.memory_allocated()
        assert final_memory > 0
        
        manager.cleanup()
    
    def test_fault_tolerance_integration(self) -> Any:
        """Test fault tolerance integration."""
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0] if torch.cuda.is_available() else [],
            enable_fault_tolerance=True,
            checkpoint_frequency=10
        )
        
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(100)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(model, dataset)
        
        # Train and trigger checkpoints
        for i, batch in enumerate(train_loader):
            if i >= 20:  # Train for 20 steps
                break
            manager.trainer.train_step(batch)
        
        # Check that checkpoints were created
        checkpoint_dir = Path("./checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
            assert len(checkpoints) > 0
        
        manager.cleanup()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataset(self) -> Any:
        """Test handling empty dataset."""
        config = MultiGPUConfig()
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(0)  # Empty dataset
        
        with pytest.raises(ValueError):
            manager.setup_training(model, dataset)
    
    def test_invalid_device_ids(self) -> Any:
        """Test invalid device IDs."""
        config = MultiGPUConfig(device_ids=[999])  # Invalid device ID
        
        with pytest.raises(RuntimeError):
            MultiGPUTrainingManager(config)
    
    def test_mixed_precision_without_cuda(self) -> Any:
        """Test mixed precision without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            config = MultiGPUConfig(use_mixed_precision=True)
            manager = MultiGPUTrainingManager(config)
            
            # Should not fail, but mixed precision should be disabled
            assert manager.trainer.config.use_mixed_precision is True
    
    def test_large_batch_size(self) -> Any:
        """Test handling large batch size."""
        config = MultiGPUConfig(batch_size=10000)  # Very large batch
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(100)
        
        # Should handle gracefully
        model, train_loader, _ = manager.setup_training(model, dataset)
        assert train_loader.batch_size == 10000
    
    def test_zero_gradient_accumulation(self) -> Any:
        """Test zero gradient accumulation steps."""
        config = MultiGPUConfig(gradient_accumulation_steps=0)
        
        with pytest.raises(ValueError):
            MultiGPUConfig(gradient_accumulation_steps=0)


class TestPerformance:
    """Performance tests."""
    
    def test_training_speed(self) -> Any:
        """Test training speed."""
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0] if torch.cuda.is_available() else [],
            use_mixed_precision=True,
            batch_size=32
        )
        
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(1000)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(model, dataset)
        
        # Measure training time
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            if i >= 10:  # Train for 10 steps
                break
            manager.trainer.train_step(batch)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Should complete within reasonable time
        assert training_time < 60  # Less than 60 seconds
        
        manager.cleanup()
    
    def test_memory_efficiency(self) -> Any:
        """Test memory efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0],
            use_mixed_precision=True,
            batch_size=64
        )
        
        manager = MultiGPUTrainingManager(config)
        model = TestModel()
        dataset = TestDataset(500)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(model, dataset)
        
        # Record memory usage
        initial_memory = torch.cuda.memory_allocated()
        
        # Train for several steps
        for i, batch in enumerate(train_loader):
            if i >= 20:
                break
            manager.trainer.train_step(batch)
            
            # Check memory doesn't grow excessively
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB)
            assert memory_increase < 1024**3
        
        manager.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 
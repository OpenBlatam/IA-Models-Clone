from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
from ..training import (
from ..models import BaseMessageModel, ModelConfig
from ..data_loader import MessageDataset
        import shutil
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Training Module
"""

    Trainer,
    TrainingManager,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG
)

class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size=100) -> Any:
        self.size = size
    
    def __len__(self) -> Any:
        return self.size
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'input_ids': torch.randint(0, 1000, (10,)),
            'attention_mask': torch.ones(10),
            'labels': torch.randint(0, 1000, (10,)),
            'original_message': f'Message {idx}',
            'message_type': 'informational'
        }

class MockModel(BaseMessageModel):
    """Mock model for testing."""
    
    def __init__(self, config) -> Any:
        super().__init__(config)
        self.linear = nn.Linear(10, 1000)  # Simple linear layer
    
    def forward(self, input_ids, attention_mask=None) -> Any:
        # Simple forward pass
        batch_size, seq_len = input_ids.shape
        return type('obj', (object,), {
            'logits': self.linear(input_ids.float())
        })
    
    def generate(self, prompt, **kwargs) -> Any:
        return f"Generated: {prompt}"
    
    def load_model(self, path) -> Any:
        pass

class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_training_config_creation(self) -> Any:
        """Test TrainingConfig creation with default values."""
        config = TrainingConfig(model_type="gpt2")
        
        assert config.model_type == "gpt2"
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 10
        assert config.warmup_steps == 1000
        assert config.max_grad_norm == 1.0
        assert config.weight_decay == 0.01
        assert config.scheduler_type == "cosine"
        assert config.gradient_accumulation_steps == 4
        assert config.use_mixed_precision is True
        assert config.fp16 is True
        assert config.use_wandb is False
        assert config.use_tensorboard is True
        assert config.experiment_name == "key_messages_training"
        assert config.save_steps == 1000
        assert config.eval_steps == 500
        assert config.save_total_limit == 3
        assert config.max_length == 512
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
    
    def test_training_config_custom_values(self) -> Any:
        """Test TrainingConfig creation with custom values."""
        config = TrainingConfig(
            model_type="bert",
            batch_size=16,
            learning_rate=2e-4,
            num_epochs=5,
            use_mixed_precision=False,
            use_wandb=True,
            experiment_name="custom_experiment"
        )
        
        assert config.model_type == "bert"
        assert config.batch_size == 16
        assert config.learning_rate == 2e-4
        assert config.num_epochs == 5
        assert config.use_mixed_precision is False
        assert config.use_wandb is True
        assert config.experiment_name == "custom_experiment"

class TestTrainer:
    """Test Trainer class."""
    
    def test_trainer_initialization(self) -> Any:
        """Test Trainer initialization."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        assert trainer.config == config
        assert trainer.model == model
        assert trainer.train_loader == train_loader
        assert trainer.val_loader == val_loader
        assert trainer.device == torch.device("cpu")
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')
        assert len(trainer.training_history) == 0
    
    def test_setup_optimizer(self) -> Any:
        """Test optimizer setup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert len(trainer.optimizer.param_groups) == 2  # One for weight decay, one without
    
    def test_setup_scheduler(self) -> Any:
        """Test scheduler setup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            scheduler_type="cosine",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_setup_scheduler_linear(self) -> Any:
        """Test linear scheduler setup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            scheduler_type="linear",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LinearLR)
    
    def test_setup_scheduler_step(self) -> Any:
        """Test step scheduler setup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            scheduler_type="step",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
    
    @patch('ml.training.SummaryWriter')
    def test_setup_experiment_tracking_tensorboard(self, mock_writer) -> Any:
        """Test TensorBoard setup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=True,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        assert trainer.writer is not None
        mock_writer.assert_called_once()
    
    @patch('ml.training.wandb')
    def test_setup_experiment_tracking_wandb(self, mock_wandb) -> Any:
        """Test Weights & Biases setup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=True
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        mock_wandb.init.assert_called_once()
    
    def test_move_batch_to_device(self) -> Any:
        """Test batch device movement."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'labels': torch.tensor([[1, 2, 3]]),
            'text': 'test'  # Non-tensor
        }
        
        device_batch = trainer._move_batch_to_device(batch)
        
        assert device_batch['input_ids'].device == torch.device("cpu")
        assert device_batch['attention_mask'].device == torch.device("cpu")
        assert device_batch['labels'].device == torch.device("cpu")
        assert device_batch['text'] == 'test'  # Non-tensor unchanged
    
    def test_compute_loss(self) -> Any:
        """Test loss computation."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'labels': torch.tensor([[1, 2, 3, 4, 5]])
        }
        
        loss = trainer._compute_loss(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_save_checkpoint(self) -> Any:
        """Test checkpoint saving."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            trainer._save_checkpoint("test_checkpoint.pt", 1)
            
            # Check if checkpoint file exists
            checkpoint_file = trainer.checkpoint_dir / "test_checkpoint.pt"
            assert checkpoint_file.exists()
    
    def test_load_checkpoint(self) -> Any:
        """Test checkpoint loading."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        # Save a checkpoint first
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer._save_checkpoint("test_checkpoint.pt", 1)
            
            # Reset trainer state
            trainer.global_step = 0
            trainer.best_val_loss = float('inf')
            trainer.training_history = []
            
            # Load checkpoint
            checkpoint_file = trainer.checkpoint_dir / "test_checkpoint.pt"
            trainer.load_checkpoint(str(checkpoint_file))
            
            assert trainer.global_step == 1
            assert trainer.best_val_loss == float('inf')  # Should be loaded from checkpoint
    
    def test_validate_epoch(self) -> bool:
        """Test validation epoch."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        val_loss = trainer._validate_epoch(0)
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0
    
    def test_get_training_summary(self) -> Optional[Dict[str, Any]]:
        """Test training summary generation."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        # Add some training history
        trainer.training_history = [
            {'epoch': 0, 'train_loss': 0.5, 'val_loss': 0.4},
            {'epoch': 1, 'train_loss': 0.3, 'val_loss': 0.2}
        ]
        trainer.best_val_loss = 0.2
        trainer.global_step = 100
        
        summary = trainer._get_training_summary()
        
        assert summary['best_val_loss'] == 0.2
        assert summary['final_epoch'] == 2
        assert summary['total_steps'] == 100
        assert len(summary['training_history']) == 2
        assert 'config' in summary
    
    def test_cleanup(self) -> Any:
        """Test trainer cleanup."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(50), batch_size=2)
        val_loader = DataLoader(MockDataset(20), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        # Mock writer and wandb run
        trainer.writer = Mock()
        trainer.wandb_run = Mock()
        
        trainer.cleanup()
        
        trainer.writer.close.assert_called_once()
        trainer.wandb_run.finish.assert_called_once()

class TestTrainingManager:
    """Test TrainingManager class."""
    
    def test_training_manager_initialization(self) -> Any:
        """Test TrainingManager initialization."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        manager = TrainingManager(config)
        
        assert manager.config == config
        assert manager.data_manager is not None
    
    @patch('ml.training.DataManager')
    @patch('ml.training.ModelFactory')
    def test_prepare_training(self, mock_factory, mock_data_manager_class) -> Any:
        """Test training preparation."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            use_tensorboard=False,
            use_wandb=False
        )
        
        manager = TrainingManager(config)
        
        # Mock data manager
        mock_data_manager = Mock()
        mock_data_manager.load_data.return_value = pd.DataFrame({
            'original_message': ['Hello world', 'Test message'],
            'message_type': ['informational', 'promotional']
        })
        mock_data_manager.create_dataset.return_value = MockDataset(10)
        mock_data_manager.get_data_loaders.return_value = (
            DataLoader(MockDataset(7), batch_size=2),
            DataLoader(MockDataset(2), batch_size=2),
            DataLoader(MockDataset(1), batch_size=2)
        )
        manager.data_manager = mock_data_manager
        
        # Mock model factory
        mock_model = MockModel(ModelConfig(model_name="test"))
        mock_factory.create_model.return_value = mock_model
        
        trainer, data_info = manager.prepare_training("test_data.csv")
        
        assert isinstance(trainer, Trainer)
        assert isinstance(data_info, dict)
        assert 'dataset_size' in data_info
        assert 'train_batches' in data_info
        assert 'val_batches' in data_info
        assert 'test_batches' in data_info
    
    @patch('ml.training.DataManager')
    @patch('ml.training.ModelFactory')
    def test_train_model(self, mock_factory, mock_data_manager_class) -> Any:
        """Test complete training pipeline."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            num_epochs=1,  # Short training for testing
            use_tensorboard=False,
            use_wandb=False
        )
        
        manager = TrainingManager(config)
        
        # Mock data manager
        mock_data_manager = Mock()
        mock_data_manager.load_data.return_value = pd.DataFrame({
            'original_message': ['Hello world', 'Test message'],
            'message_type': ['informational', 'promotional']
        })
        mock_data_manager.create_dataset.return_value = MockDataset(10)
        mock_data_manager.get_data_loaders.return_value = (
            DataLoader(MockDataset(7), batch_size=2),
            DataLoader(MockDataset(2), batch_size=2),
            DataLoader(MockDataset(1), batch_size=2)
        )
        manager.data_manager = mock_data_manager
        
        # Mock model factory
        mock_model = MockModel(ModelConfig(model_name="test"))
        mock_factory.create_model.return_value = mock_model
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            'best_val_loss': 0.1,
            'final_epoch': 1,
            'total_steps': 10,
            'training_history': []
        }
        mock_trainer.checkpoint_dir = Path("./test_checkpoints")
        mock_trainer.checkpoint_dir.mkdir(exist_ok=True)
        
        with patch('ml.training.Trainer', return_value=mock_trainer):
            results = manager.train_model("test_data.csv")
        
        assert isinstance(results, dict)
        assert 'training_summary' in results
        assert 'data_info' in results
        assert 'model_path' in results
        
        # Cleanup
        if Path("./test_checkpoints").exists():
            shutil.rmtree("./test_checkpoints")

class TestDefaultTrainingConfig:
    """Test default training configuration."""
    
    def test_default_training_config(self) -> Any:
        """Test DEFAULT_TRAINING_CONFIG."""
        assert DEFAULT_TRAINING_CONFIG.model_type == "gpt2"
        assert DEFAULT_TRAINING_CONFIG.model_name == "gpt2"
        assert DEFAULT_TRAINING_CONFIG.batch_size == 16
        assert DEFAULT_TRAINING_CONFIG.learning_rate == 1e-4
        assert DEFAULT_TRAINING_CONFIG.num_epochs == 5
        assert DEFAULT_TRAINING_CONFIG.gradient_accumulation_steps == 4
        assert DEFAULT_TRAINING_CONFIG.use_mixed_precision is True
        assert DEFAULT_TRAINING_CONFIG.use_tensorboard is True
        assert DEFAULT_TRAINING_CONFIG.use_wandb is False
        assert DEFAULT_TRAINING_CONFIG.experiment_name == "key_messages_baseline"

class TestTrainingIntegration:
    """Integration tests for training."""
    
    def test_end_to_end_training_short(self) -> Any:
        """Test end-to-end training with minimal epochs."""
        config = TrainingConfig(
            model_type="gpt2",
            device="cpu",
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(10), batch_size=2)
        val_loader = DataLoader(MockDataset(5), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        # Run training
        results = trainer.train()
        
        assert isinstance(results, dict)
        assert 'best_val_loss' in results
        assert 'final_epoch' in results
        assert 'total_steps' in results
        assert 'training_history' in results
        assert 'config' in results
        assert results['final_epoch'] == 1
    
    def test_mixed_precision_training(self) -> Any:
        """Test mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        config = TrainingConfig(
            model_type="gpt2",
            device="cuda",
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            use_mixed_precision=True,
            use_tensorboard=False,
            use_wandb=False
        )
        
        model = MockModel(ModelConfig(model_name="test"))
        train_loader = DataLoader(MockDataset(10), batch_size=2)
        val_loader = DataLoader(MockDataset(5), batch_size=2)
        
        trainer = Trainer(config, model, train_loader, val_loader)
        
        # Check that scaler is initialized
        assert trainer.scaler is not None
        
        # Run training
        results = trainer.train()
        
        assert isinstance(results, dict)
        assert 'best_val_loss' in results

match __name__:
    case "__main__":
    pytest.main([__file__]) 
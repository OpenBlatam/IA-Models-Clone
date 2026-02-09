"""
Tests for Training Module
=========================
Tests for training loops and utilities
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..training_module import TrainingLoop, PersonalityTrainingLoop
from ..callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from ..loss_functions import PersonalityTraitLoss
from ..optimizers import create_optimizer


class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (20,)),
            "attention_mask": torch.ones(20),
            "personality_labels": torch.rand(5)
        }


class DummyModel(nn.Module):
    """Dummy model for testing"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 5)
    
    def forward(self, input_ids, attention_mask=None):
        # Simplified forward
        return {
            "openness": torch.sigmoid(self.linear(input_ids.mean(dim=1))),
            "conscientiousness": torch.sigmoid(self.linear(input_ids.mean(dim=1))),
            "extraversion": torch.sigmoid(self.linear(input_ids.mean(dim=1))),
            "agreeableness": torch.sigmoid(self.linear(input_ids.mean(dim=1))),
            "neuroticism": torch.sigmoid(self.linear(input_ids.mean(dim=1)))
        }


class TestTrainingLoop:
    """Tests for training loop"""
    
    def test_loop_initialization(self):
        """Test training loop initialization"""
        model = DummyModel()
        dataset = DummyDataset()
        train_loader = DataLoader(dataset, batch_size=2)
        
        loop = PersonalityTrainingLoop(
            model=model,
            train_loader=train_loader
        )
        assert loop is not None
        assert loop.device is not None
    
    def test_training_step(self):
        """Test training step"""
        model = DummyModel()
        dataset = DummyDataset(size=4)
        train_loader = DataLoader(dataset, batch_size=2)
        
        loop = PersonalityTrainingLoop(
            model=model,
            train_loader=train_loader
        )
        
        # Test one epoch
        metrics = loop.train_epoch(0)
        assert "loss" in metrics
        assert metrics["loss"] >= 0


class TestCallbacks:
    """Tests for callbacks"""
    
    def test_early_stopping_callback(self):
        """Test early stopping callback"""
        callback = EarlyStoppingCallback(
            monitor="val_loss",
            patience=3,
            min_delta=0.001
        )
        
        callback.on_train_begin()
        
        # Simulate improving loss
        logs = {"val_loss": 0.5}
        should_stop = callback.on_epoch_end(0, logs)
        assert not should_stop
        
        # Simulate no improvement
        for i in range(4):
            logs = {"val_loss": 0.6}  # Worse than best
            should_stop = callback.on_epoch_end(i + 1, logs)
        
        # Should trigger after patience
        assert should_stop
    
    def test_checkpoint_callback(self):
        """Test checkpoint callback"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpointCallback(
                filepath=os.path.join(tmpdir, "test_model.pt"),
                monitor="val_loss",
                save_best_only=True
            )
            
            model = DummyModel()
            logs = {"val_loss": 0.5}
            callback.on_epoch_end(0, logs, model)
            
            # Check if file was created
            assert os.path.exists(os.path.join(tmpdir, "test_model_epoch_0.pt"))


class TestLossFunctions:
    """Tests for loss functions"""
    
    def test_personality_trait_loss(self):
        """Test personality trait loss"""
        loss_fn = PersonalityTraitLoss()
        
        predictions = {
            "openness": torch.tensor([[0.7], [0.6]]),
            "conscientiousness": torch.tensor([[0.8], [0.7]]),
            "extraversion": torch.tensor([[0.6], [0.5]]),
            "agreeableness": torch.tensor([[0.9], [0.8]]),
            "neuroticism": torch.tensor([[0.4], [0.3]])
        }
        
        targets = torch.tensor([[0.7, 0.8, 0.6, 0.9, 0.4], [0.6, 0.7, 0.5, 0.8, 0.3]])
        
        loss = loss_fn(predictions, targets)
        assert loss.item() >= 0


class TestOptimizers:
    """Tests for optimizers"""
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        model = DummyModel()
        
        optimizer = create_optimizer(
            "adam",
            model.parameters(),
            learning_rate=1e-3
        )
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)





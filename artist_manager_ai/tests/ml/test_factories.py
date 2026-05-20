"""
Tests for Factories
===================

Unit tests for factory classes.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.factories import ModelFactory, TrainerFactory, DataFactory
from ml.models import EventDurationPredictor


class TestModelFactory:
    """Tests for ModelFactory."""
    
    def test_create_model(self):
        """Test model creation."""
        factory = ModelFactory()
        model = factory.create(
            "event_duration",
            config={"input_dim": 32, "hidden_dims": [64, 32]}
        )
        assert model is not None
        assert isinstance(model, EventDurationPredictor)
    
    def test_list_models(self):
        """Test listing available models."""
        models = ModelFactory.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "event_duration" in models
    
    def test_register_model(self):
        """Test model registration."""
        class CustomModel(EventDurationPredictor):
            pass
        
        ModelFactory.register_model("custom", CustomModel)
        assert "custom" in ModelFactory.MODEL_REGISTRY


class TestTrainerFactory:
    """Tests for TrainerFactory."""
    
    def test_create_trainer(self):
        """Test trainer creation."""
        factory = TrainerFactory()
        model = EventDurationPredictor(input_dim=32)
        
        # Create dummy dataloaders
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 32),
            torch.randn(10, 1)
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        trainer = factory.create(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=torch.nn.MSELoss(),
            optimizer_type="adam"
        )
        assert trainer is not None


class TestDataFactory:
    """Tests for DataFactory."""
    
    def test_create_dataset(self):
        """Test dataset creation."""
        factory = DataFactory()
        
        events = [
            {
                "type": "concert",
                "start_time": "2024-01-01T20:00:00",
                "end_time": "2024-01-01T22:00:00"
            }
        ]
        
        dataset = factory.create_dataset("event", events)
        assert dataset is not None
        assert len(dataset) > 0





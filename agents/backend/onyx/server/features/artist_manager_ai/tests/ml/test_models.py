"""
Tests for ML Models
===================

Unit tests for ML models.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models import EventDurationPredictor, RoutineCompletionPredictor, OptimalTimePredictor


class TestEventDurationPredictor:
    """Tests for EventDurationPredictor."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = EventDurationPredictor(
            input_dim=32,
            hidden_dims=[64, 32],
            dropout_rate=0.2
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = EventDurationPredictor(input_dim=32)
        x = torch.randn(1, 32)
        output = model(x)
        assert output.shape == (1, 1)
        assert output.item() >= 0  # Duration should be non-negative
    
    def test_prediction(self):
        """Test prediction method."""
        model = EventDurationPredictor(input_dim=32)
        x = torch.randn(1, 32)
        result = model.predict(x)
        assert "predicted_duration" in result
        assert result["predicted_duration"] >= 0


class TestRoutineCompletionPredictor:
    """Tests for RoutineCompletionPredictor."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = RoutineCompletionPredictor(
            input_dim=16,
            lstm_hidden=32,
            lstm_layers=1
        )
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = RoutineCompletionPredictor(input_dim=16)
        x = torch.randn(1, 7, 16)  # [batch, seq_len, features]
        output = model(x)
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1  # Probability


class TestOptimalTimePredictor:
    """Tests for OptimalTimePredictor."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = OptimalTimePredictor(
            input_dim=24,
            hidden_dim=64,
            num_hours=24
        )
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = OptimalTimePredictor(input_dim=24)
        x = torch.randn(1, 24)
        output = model(x)
        assert output.shape == (1, 24)  # 24 hour classes
    
    def test_prediction(self):
        """Test prediction method."""
        model = OptimalTimePredictor(input_dim=24)
        x = torch.randn(1, 24)
        result = model.predict(x)
        assert "optimal_hour" in result
        assert 0 <= result["optimal_hour"] <= 23


if __name__ == "__main__":
    pytest.main([__file__])





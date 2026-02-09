"""
Routine Completion Predictor Model
===================================

PyTorch model for predicting routine completion rate.
Refactored to inherit from BaseModel following best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import logging

from ..base import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class RoutineCompletionPredictorConfig(ModelConfig):
    """Configuration for RoutineCompletionPredictor."""
    input_dim: int
    output_dim: int = 1
    lstm_hidden: int = 64
    lstm_layers: int = 2
    fc_dims: List[int] = None
    dropout_rate: float = 0.2
    device: str = "auto"
    dtype: str = "float32"
    
    def __post_init__(self):
        """Set default fc_dims if not provided."""
        if self.fc_dims is None:
            self.fc_dims = [128, 64]


class RoutineCompletionPredictor(BaseModel):
    """
    Neural network for predicting routine completion rate.
    
    Architecture:
    - LSTM layers for temporal pattern recognition
    - Fully connected layers for final prediction
    - Output: completion probability (0-1)
    
    Best Practices:
    - LSTM for sequential/temporal data
    - Proper initialization
    - Dropout for regularization
    - Sigmoid activation for probability output
    """
    
    def __init__(
        self,
        config: Optional[RoutineCompletionPredictorConfig] = None,
        **kwargs
    ):
        """
        Initialize the model.
        
        Args:
            config: Model configuration (optional)
            **kwargs: Alternative way to pass config parameters
        """
        # Handle both config object and kwargs
        if config is None:
            config_dict = {
                "input_dim": kwargs.get("input_dim", 16),
                "output_dim": 1,
                "lstm_hidden": kwargs.get("lstm_hidden", 64),
                "lstm_layers": kwargs.get("lstm_layers", 2),
                "fc_dims": kwargs.get("fc_dims", [128, 64]),
                "dropout_rate": kwargs.get("dropout_rate", 0.2),
                "device": kwargs.get("device", "auto"),
                "dtype": kwargs.get("dtype", "float32")
            }
            config = RoutineCompletionPredictorConfig(**config_dict)
        
        # Initialize base class
        super().__init__(config)
        
        # Store model-specific config
        self.lstm_hidden = config.lstm_hidden
        self.lstm_layers = config.lstm_layers
        self.fc_dims = config.fc_dims
        self.dropout_rate = config.dropout_rate
        
        # Build LSTM
        self.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.lstm_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Build fully connected layers
        self.fc_layers = self._build_fc_layers()
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _build_fc_layers(self) -> nn.Sequential:
        """
        Build fully connected layers.
        
        Returns:
            Sequential FC layers
        """
        layers = []
        prev_dim = self.lstm_hidden
        
        for fc_dim in self.fc_dims:
            layers.append(nn.Linear(prev_dim, fc_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = fc_dim
        
        # Output layer with sigmoid for probability
        layers.append(nn.Linear(prev_dim, self.config.output_dim))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights using proper initialization.
        
        Best Practice: Proper initialization for LSTM and FC layers.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weight initialization
                    nn.init.xavier_uniform_(param)
                else:
                    # FC layer initialization
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        
        Returns:
            Completion probability [batch_size, 1]
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, features], got {x.dim()}D")
        
        if x.size(2) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {x.size(2)}"
            )
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state from last layer
        last_hidden = hidden[-1]  # [batch_size, lstm_hidden]
        
        # Fully connected layers
        output = self.fc_layers(last_hidden)
        
        return output
    
    def predict(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make prediction.
        
        Args:
            x: Input features [seq_len, input_dim] or [batch, seq_len, input_dim]
            return_confidence: Whether to return confidence
            **kwargs: Additional arguments
        
        Returns:
            Prediction dictionary with:
            - completion_probability: Probability of completion (0-1)
            - predicted_completion_rate: Same as completion_probability
            - confidence: Confidence score (if requested)
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Move to correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            prediction = self.forward(x)
            
            # Extract probability
            if prediction.dim() > 1:
                prob = prediction.squeeze().item()
            else:
                prob = prediction.item()
            
            # Clamp to [0, 1]
            prob = max(0.0, min(1.0, prob))
            
            result: Dict[str, Any] = {
                "completion_probability": float(prob),
                "predicted_completion_rate": float(prob)
            }
            
            if return_confidence:
                # Confidence based on prediction certainty
                # Higher confidence when prediction is far from 0.5
                confidence = abs(prob - 0.5) * 2.0
                result["confidence"] = float(confidence)
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        base_config = super().get_config()
        base_config.update({
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "fc_dims": self.fc_dims,
            "dropout_rate": self.dropout_rate
        })
        return base_config

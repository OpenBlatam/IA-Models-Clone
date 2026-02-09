"""
Event Duration Predictor Model
===============================

PyTorch model for predicting event duration.
Refactored to inherit from BaseModel following best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import logging

from ..base import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class EventDurationPredictorConfig(ModelConfig):
    """Configuration for EventDurationPredictor."""
    input_dim: int
    output_dim: int = 1
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    device: str = "auto"
    dtype: str = "float32"
    
    def __post_init__(self):
        """Set default hidden_dims if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class EventDurationPredictor(BaseModel):
    """
    Neural network for predicting event duration.
    
    Architecture:
    - Input features: event type, day of week, month, location type, etc.
    - Hidden layers with dropout for regularization
    - Batch normalization for stability
    - Output: predicted duration in hours (non-negative)
    
    Best Practices:
    - Proper weight initialization (Xavier uniform)
    - Batch normalization for training stability
    - Dropout for regularization
    - ReLU activation for non-linearity
    """
    
    def __init__(self, config: Optional[EventDurationPredictorConfig] = None, **kwargs):
        """
        Initialize the model.
        
        Args:
            config: Model configuration (optional)
            **kwargs: Alternative way to pass config parameters
        """
        # Handle both config object and kwargs
        if config is None:
            # Create config from kwargs or defaults
            config_dict = {
                "input_dim": kwargs.get("input_dim", 32),
                "output_dim": 1,
                "hidden_dims": kwargs.get("hidden_dims", [128, 64, 32]),
                "dropout_rate": kwargs.get("dropout_rate", 0.2),
                "use_batch_norm": kwargs.get("use_batch_norm", True),
                "device": kwargs.get("device", "auto"),
                "dtype": kwargs.get("dtype", "float32")
            }
            config = EventDurationPredictorConfig(**config_dict)
        
        # Initialize base class
        super().__init__(config)
        
        # Store model-specific config
        self.hidden_dims = config.hidden_dims
        self.dropout_rate = config.dropout_rate
        self.use_batch_norm = config.use_batch_norm
        
        # Build network layers
        self.network = self._build_network()
        
        # Initialize weights using best practices
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _build_network(self) -> nn.Sequential:
        """
        Build the neural network architecture.
        
        Returns:
            Sequential network
        """
        layers = []
        prev_dim = self.config.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (if enabled)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout for regularization
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (regression)
        layers.append(nn.Linear(prev_dim, self.config.output_dim))
        layers.append(nn.ReLU())  # Ensure non-negative duration
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights using Xavier uniform initialization.
        
        Best Practice: Proper weight initialization prevents vanishing/exploding gradients.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                # Standard initialization for batch norm
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Predicted duration [batch_size, 1]
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        if x.size(1) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {x.size(1)}"
            )
        
        # Forward through network
        return self.network(x)
    
    def predict(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make prediction with optional confidence estimation.
        
        Args:
            x: Input features [batch_size, input_dim] or [input_dim]
            return_confidence: Whether to return confidence score
            **kwargs: Additional arguments (unused, for interface compatibility)
        
        Returns:
            Prediction dictionary with:
            - predicted_duration: Predicted duration in hours
            - confidence: Confidence score (if requested)
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Move to correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            prediction = self.forward(x)
            
            # Extract scalar value
            if prediction.dim() > 1:
                duration = prediction.squeeze().item()
            else:
                duration = prediction.item()
            
            # Ensure non-negative
            duration = max(0.0, duration)
            
            result: Dict[str, Any] = {
                "predicted_duration": float(duration)
            }
            
            if return_confidence:
                # Simple confidence estimation
                # In production, use ensemble methods or MC dropout
                result["confidence"] = 0.8  # Placeholder
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        base_config = super().get_config()
        base_config.update({
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        })
        return base_config

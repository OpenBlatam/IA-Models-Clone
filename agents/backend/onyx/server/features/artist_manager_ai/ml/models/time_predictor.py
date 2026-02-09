"""
Optimal Time Predictor Model
=============================

PyTorch model for predicting optimal event time.
Refactored to inherit from BaseModel following best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import logging

from ..base import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class OptimalTimePredictorConfig(ModelConfig):
    """Configuration for OptimalTimePredictor."""
    input_dim: int
    output_dim: int = 24  # 24 hours
    hidden_dim: int = 128
    num_hours: int = 24
    num_heads: int = 4
    dropout_rate: float = 0.2
    device: str = "auto"
    dtype: str = "float32"


class OptimalTimePredictor(BaseModel):
    """
    Neural network for predicting optimal event time.
    
    Architecture:
    - Feature extraction layers
    - Multi-head attention for feature importance
    - Classification head for 24 hour classes
    
    Best Practices:
    - Attention mechanism for feature importance
    - Proper initialization
    - Dropout for regularization
    - Softmax for probability distribution over hours
    """
    
    def __init__(
        self,
        config: Optional[OptimalTimePredictorConfig] = None,
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
                "input_dim": kwargs.get("input_dim", 24),
                "output_dim": kwargs.get("num_hours", 24),
                "hidden_dim": kwargs.get("hidden_dim", 128),
                "num_hours": kwargs.get("num_hours", 24),
                "num_heads": kwargs.get("num_heads", 4),
                "dropout_rate": kwargs.get("dropout_rate", 0.2),
                "device": kwargs.get("device", "auto"),
                "dtype": kwargs.get("dtype", "float32")
            }
            config = OptimalTimePredictorConfig(**config_dict)
        
        # Initialize base class
        super().__init__(config)
        
        # Store model-specific config
        self.hidden_dim = config.hidden_dim
        self.num_hours = config.num_hours
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        
        # Feature extraction
        self.feature_extractor = self._build_feature_extractor()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Classification head
        self.classifier = self._build_classifier()
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _build_feature_extractor(self) -> nn.Sequential:
        """
        Build feature extraction layers.
        
        Returns:
            Sequential feature extractor
        """
        return nn.Sequential(
            nn.Linear(self.config.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
    
    def _build_classifier(self) -> nn.Sequential:
        """
        Build classification head.
        
        Returns:
            Sequential classifier
        """
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_hours)
        )
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights using proper initialization.
        
        Best Practice: Xavier uniform for linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Hour logits [batch_size, num_hours]
        
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
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Add sequence dimension for attention [batch, 1, hidden]
        features_seq = features.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.attention(features_seq, features_seq, features_seq)
        attn_out = attn_out.squeeze(1)  # [batch, hidden]
        
        # Classification
        logits = self.classifier(attn_out)
        
        return logits
    
    def predict(
        self,
        x: torch.Tensor,
        return_probabilities: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make prediction.
        
        Args:
            x: Input features [input_dim] or [batch, input_dim]
            return_probabilities: Whether to return hour probabilities
            **kwargs: Additional arguments
        
        Returns:
            Prediction dictionary with:
            - optimal_hour: Predicted optimal hour (0-23)
            - confidence: Confidence score
            - hour_probabilities: Probability distribution (if requested)
        
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
            logits = self.forward(x)
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted hour
            predicted_hour = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_hour].item()
            
            result: Dict[str, Any] = {
                "optimal_hour": int(predicted_hour),
                "confidence": float(confidence)
            }
            
            if return_probabilities:
                result["hour_probabilities"] = probabilities.squeeze().cpu().tolist()
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        base_config = super().get_config()
        base_config.update({
            "hidden_dim": self.hidden_dim,
            "num_hours": self.num_hours,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return base_config

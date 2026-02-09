"""
Deep Genre Classifier Module

Implements deep neural network for genre classification.
"""

from typing import List
import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DeepGenreClassifier(nn.Module):
    """
    Deep Neural Network for Genre Classification.
    Multi-layer architecture with residual connections.
    
    Args:
        input_size: Input feature size.
        num_genres: Number of genre classes.
        hidden_layers: List of hidden layer sizes.
        dropout_rate: Dropout probability.
        use_batch_norm: If True, use batch normalization.
        use_residual: If True, use residual connections.
    """
    
    def __init__(
        self,
        input_size: int = 169,
        num_genres: int = 10,
        hidden_layers: List[int] = [512, 512, 256, 256, 128, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.input_size = input_size
        self.num_genres = num_genres
        self.use_residual = use_residual
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        self.input_bn = nn.BatchNorm1d(hidden_layers[0]) if use_batch_norm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        for i in range(len(hidden_layers) - 1):
            # Main layer
            self.hidden_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            )
            
            # Batch normalization
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            else:
                self.bn_layers.append(nn.Identity())
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            
            # Residual connection (if dimensions match)
            if use_residual and hidden_layers[i] == hidden_layers[i + 1]:
                self.residual_layers.append(nn.Identity())
            else:
                self.residual_layers.append(
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1])
                    if use_residual else nn.Identity()
                )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], num_genres)
        
        # Initialize weights
        self._initialize_weights()
        logger.debug(f"Initialized DeepGenreClassifier with input_size={input_size}, num_genres={num_genres}")
    
    def _initialize_weights(self):
        """Initialize weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming for ReLU, Xavier for others
                if hasattr(module, 'activation') and module.activation == 'relu':
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_size]
        
        Returns:
            Genre logits [batch_size, num_genres]
        """
        # Input layer
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # Hidden layers with residual connections
        for i, (hidden, bn, dropout, residual) in enumerate(
            zip(self.hidden_layers, self.bn_layers, self.dropout_layers, self.residual_layers)
        ):
            residual_input = x
            x = hidden(x)
            x = bn(x)
            x = torch.relu(x)
            x = dropout(x)
            
            if self.use_residual:
                x = x + residual(residual_input)
        
        # Output layer
        x = self.output_layer(x)
        return x




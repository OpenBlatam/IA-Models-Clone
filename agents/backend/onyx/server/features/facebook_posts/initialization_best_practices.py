from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Weight Initialization Best Practices
Comprehensive best practices for weight initialization in deep learning models.
"""



class ArchitectureType(Enum):
    """Types of neural network architectures."""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"
    RESIDUAL = "residual"
    ATTENTION = "attention"


@dataclass
class InitializationBestPractices:
    """Best practices for weight initialization."""
    architecture_type: ArchitectureType = ArchitectureType.FEEDFORWARD
    activation_function: str = "relu"
    input_dimension: int = 100
    hidden_dimension: int = 50
    output_dimension: int = 10
    num_layers: int = 3
    use_bias: bool = True
    dropout_rate: float = 0.1


class FeedforwardInitializer:
    """Best practices for feedforward neural networks."""
    
    @staticmethod
    def initialize_feedforward_weights(module: nn.Modinear, 
                                     activation: str = "relu") -> None:
        """Initialize weights for feedforward networks."""
        if isinstance(module, nn.Linear):
            # Use He initialization for ReLU activations
            if activation.lower() in ["relu", "leaky_relu"]:
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # Use Xavier initialization for sigmoid/tanh activations
            elif activation.lower() in ["sigmoid", "tanh"]:
                init.xavier_normal_(module.weight, gain=1.0)
            # Use orthogonal initialization for other activations
            else:
                init.orthogonal_(module.weight, gain=1.0)
            
            if module.bias is not None:
                init.zeros_(module.bias)


class ConvolutionalInitializer:
    """Best practices for convolutional neural networks."""
    
    @staticmethod
    def initialize_convolutional_weights(module: nn.Module, 
                                       activation: str = "relu") -> None:
        """Initialize weights for convolutional networks."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Use He initialization for ReLU activations
            if activation.lower() in ["relu", "leaky_relu"]:
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            # Use Xavier initialization for sigmoid/tanh activations
            elif activation.lower() in ["sigmoid", "tanh"]:
                init.xavier_normal_(module.weight, gain=1.0)
            # Use orthogonal initialization for other activations
            else:
                init.orthogonal_(module.weight, gain=1.0)
            
            if module.bias is not None:
                init.zeros_(module.bias)


class RecurrentInitializer:
    """Best practices for recurrent neural networks."""
    
    @staticmethod
    def initialize_recurrent_weights(module: nn.Module) -> None:
        """Initialize weights for recurrent networks."""
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    # Use orthogonal initialization for recurrent weights
                    init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    # Initialize bias with zeros
                    init.zeros_(param)
                    
                    # Special initialization for LSTM forget gate bias
                    if isinstance(module, nn.LSTM) and 'bias_ih' in name:
                        # Set forget gate bias to 1 for better gradient flow
                        n = module.hidden_size
                        param.data[n:2*n].fill_(1.0)


class TransformerInitializer:
    """Best practices for transformer architectures."""
    
    @staticmethod
    def initialize_transformer_weights(module: nn.Module) -> None:
        """Initialize weights for transformer networks."""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for transformer linear layers
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            # Use normal initialization for embeddings
            init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer norm with ones and zeros
            init.ones_(module.weight)
            init.zeros_(module.bias)


class ResidualInitializer:
    """Best practices for residual networks."""
    
    @staticmethod
    def initialize_residual_weights(module: nn.Module, 
                                  activation: str = "relu") -> None:
        """Initialize weights for residual networks."""
        if isinstance(module, nn.Linear):
            # Use He initialization for residual connections
            if activation.lower() in ["relu", "leaky_relu"]:
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            else:
                init.xavier_normal_(module.weight, gain=1.0)
            
            if module.bias is not None:
                init.zeros_(module.bias)
        
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Use He initialization for convolutional layers
            if activation.lower() in ["relu", "leaky_relu"]:
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            else:
                init.xavier_normal_(module.weight, gain=1.0)
            
            if module.bias is not None:
                init.zeros_(module.bias)


class AttentionInitializer:
    """Best practices for attention mechanisms."""
    
    @staticmethod
    def initialize_attention_weights(module: nn.Module) -> None:
        """Initialize weights for attention mechanisms."""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for attention layers
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)


class AdvancedInitializationManager:
    """Advanced initialization manager with best practices."""
    
    def __init__(self, architecture_type: ArchitectureType, 
                 activation_function: str = "relu"):
        
    """__init__ function."""
self.architecture_type = architecture_type
        self.activation_function = activation_function
        
        # Initialize appropriate initializer
        self.initializer = self._get_initializer()
    
    def _get_initializer(self) -> Callable:
        """Get appropriate initializer for architecture type."""
        if self.architecture_type == ArchitectureType.FEEDFORWARD:
            return lambda module: FeedforwardInitializer.initialize_feedforward_weights(
                module, self.activation_function
            )
        elif self.architecture_type == ArchitectureType.CONVOLUTIONAL:
            return lambda module: ConvolutionalInitializer.initialize_convolutional_weights(
                module, self.activation_function
            )
        elif self.architecture_type == ArchitectureType.RECURRENT:
            return RecurrentInitializer.initialize_recurrent_weights
        elif self.architecture_type == ArchitectureType.TRANSFORMER:
            return TransformerInitializer.initialize_transformer_weights
        elif self.architecture_type == ArchitectureType.RESIDUAL:
            return lambda module: ResidualInitializer.initialize_residual_weights(
                module, self.activation_function
            )
        elif self.architecture_type == ArchitectureType.ATTENTION:
            return AttentionInitializer.initialize_attention_weights
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture_type}")
    
    def initialize_model(self, model: nn.Module) -> None:
        """Initialize all weights in the model using best practices."""
        for module in model.modules():
            self.initializer(module)


class FeedforwardNetwork(nn.Module):
    """Feedforward neural network with proper initialization."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, activation: str = "relu", dropout_rate: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using best practices."""
        initializer = AdvancedInitializationManager(
            ArchitectureType.FEEDFORWARD, self.activation
        )
        initializer.initialize_model(self)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper activation functions."""
        hidden = input_tensor
        
        for i, layer in enumerate(self.layers[:-1]):
            hidden = layer(hidden)
            
            # Apply activation
            if self.activation.lower() == "relu":
                hidden = F.relu(hidden)
            elif self.activation.lower() == "leaky_relu":
                hidden = F.leaky_relu(hidden)
            elif self.activation.lower() == "gelu":
                hidden = F.gelu(hidden)
            elif self.activation.lower() == "tanh":
                hidden = torch.tanh(hidden)
            elif self.activation.lower() == "sigmoid":
                hidden = torch.sigmoid(hidden)
            
            # Apply dropout (except for last layer)
            if i < len(self.layers) - 2:
                hidden = self.dropout(hidden)
        
        # Output layer
        output = self.layers[-1](hidden)
        
        return output


class ConvolutionalNetwork(nn.Module):
    """Convolutional neural network with proper initialization."""
    
    def __init__(self, input_channels: int, num_classes: int, 
                 activation: str = "relu", dropout_rate: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using best practices."""
        initializer = AdvancedInitializationManager(
            ArchitectureType.CONVOLUTIONAL, self.activation
        )
        initializer.initialize_model(self)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper activation functions."""
        # Convolutional layers
        hidden = self.conv1(input_tensor)
        hidden = self.batch_norm1(hidden)
        if self.activation.lower() == "relu":
            hidden = F.relu(hidden)
        hidden = self.pool(hidden)
        
        hidden = self.conv2(hidden)
        hidden = self.batch_norm2(hidden)
        if self.activation.lower() == "relu":
            hidden = F.relu(hidden)
        hidden = self.pool(hidden)
        
        hidden = self.conv3(hidden)
        hidden = self.batch_norm3(hidden)
        if self.activation.lower() == "relu":
            hidden = F.relu(hidden)
        hidden = self.pool(hidden)
        
        # Flatten
        hidden = hidden.view(hidden.size(0), -1)
        
        # Fully connected layers
        hidden = self.fc1(hidden)
        if self.activation.lower() == "relu":
            hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        
        output = self.fc2(hidden)
        
        return output


class RecurrentNetwork(nn.Module):
    """Recurrent neural network with proper initialization."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 bidirectional: bool = True, dropout_rate: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(output_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using best practices."""
        initializer = AdvancedInitializationManager(ArchitectureType.RECURRENT)
        initializer.initialize_model(self)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(input_tensor)
        
        # Use last output
        if self.bidirectional:
            # Concatenate forward and backward outputs
            output = torch.cat((lstm_out[:, -1, :self.hidden_size], 
                              lstm_out[:, 0, self.hidden_size:]), dim=1)
        else:
            output = lstm_out[:, -1, :]
        
        # Final linear layer
        output = self.fc(output)
        
        return output


def demonstrate_initialization_best_practices():
    """Demonstrate weight initialization best practices."""
    print("Weight Initialization Best Practices")
    print("=" * 50)
    
    # Test different architectures
    architectures = [
        (ArchitectureType.FEEDFORWARD, "relu"),
        (ArchitectureType.CONVOLUTIONAL, "relu"),
        (ArchitectureType.RECURRENT, "tanh"),
        (ArchitectureType.TRANSFORMER, "gelu"),
        (ArchitectureType.RESIDUAL, "relu")
    ]
    
    results = {}
    
    for arch_type, activation in architectures:
        print(f"\nTesting {arch_type.value.upper()} with {activation} activation:")
        
        try:
            # Create appropriate model
            if arch_type == ArchitectureType.FEEDFORWARD:
                model = FeedforwardNetwork(100, 64, 10, activation=activation)
            elif arch_type == ArchitectureType.CONVOLUTIONAL:
                model = ConvolutionalNetwork(3, 10, activation=activation)
            elif arch_type == ArchitectureType.RECURRENT:
                model = RecurrentNetwork(50, 32)
            else:
                # Create a simple linear model for other architectures
                model = nn.Linear(100, 10)
            
            # Initialize weights
            initializer = AdvancedInitializationManager(arch_type, activation)
            initializer.initialize_model(model)
            
            # Analyze weight statistics
            weight_stats = analyze_weight_statistics(model)
            
            print(f"  Mean weight: {weight_stats['mean']:.6f}")
            print(f"  Weight std: {weight_stats['std']:.6f}")
            print(f"  Weight range: [{weight_stats['min']:.6f}, {weight_stats['max']:.6f}]")
            
            results[arch_type.value] = {
                'activation': activation,
                'weight_stats': weight_stats,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[arch_type.value] = {
                'activation': activation,
                'error': str(e),
                'success': False
            }
    
    return results


def analyze_weight_statistics(model: nn.Module) -> Dict[str, float]:
    """Analyze weight statistics for a model."""
    all_weights = []
    
    for param in model.parameters():
        if param.dim() > 1:  # Only analyze weight matrices
            all_weights.append(param.data.flatten())
    
    if not all_weights:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    all_weights = torch.cat(all_weights)
    
    return {
        'mean': all_weights.mean().item(),
        'std': all_weights.std().item(),
        'min': all_weights.min().item(),
        'max': all_weights.max().item()
    }


if __name__ == "__main__":
    # Demonstrate initialization best practices
    results = demonstrate_initialization_best_practices()
    print("\nWeight initialization best practices demonstration completed!") 
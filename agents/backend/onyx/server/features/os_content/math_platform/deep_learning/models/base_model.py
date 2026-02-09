from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Base Model Class
Foundation class for all model architectures in the modular deep learning system.
"""


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base configuration for models."""
    # Model parameters
    model_name: str = "base_model"
    model_type: str = "base"
    input_size: int = 784
    output_size: int = 10
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    
    # Architecture parameters
    activation: str = "relu"  # relu, tanh, sigmoid, gelu, swish
    dropout_rate: float = 0.1
    batch_norm: bool = True
    layer_norm: bool = False
    
    # Initialization parameters
    weight_init: str = "xavier"  # xavier, kaiming, normal, uniform
    bias_init: str = "zeros"
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Device parameters
    device: str = "auto"  # auto, cpu, cuda
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'layer_norm': self.layer_norm,
            'weight_init': self.weight_init,
            'bias_init': self.bias_init,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clipping': self.gradient_clipping,
            'max_grad_norm': self.max_grad_norm,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseModel(nn.Module, ABC):
    """Base model class for all neural network architectures."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.device = self._setup_device()
        self.activation_fn = self._get_activation_function()
        
        # Initialize model components
        self._build_model()
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized {self.config.model_name} on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup device for model."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _get_activation_function(self) -> nn.Module:
        """Get activation function based on config."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is the same as Swish
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        
        if self.config.activation not in activation_map:
            logger.warning(f"Unknown activation function: {self.config.activation}. Using ReLU.")
            return activation_map['relu']
        
        return activation_map[self.config.activation]
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif self.config.weight_init == "normal":
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                elif self.config.weight_init == "uniform":
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                
                if module.bias is not None:
                    if self.config.bias_init == "zeros":
                        nn.init.zeros_(module.bias)
                    elif self.config.bias_init == "normal":
                        nn.init.normal_(module.bias, mean=0, std=0.01)
            
            elif isinstance(module, nn.Conv2d):
                if self.config.weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        logger.info(f"Initialized weights using {self.config.weight_init} initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Forward method must be implemented by subclasses")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'model_name': self.config.model_name,
            'model_type': self.config.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'config': self.config.to_dict()
        }
        
        return model_info
    
    def save_model(self, filepath: str, save_config: bool = True):
        """Save model and optionally config."""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'config': self.config.to_dict() if save_config else None
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, strict: bool = True):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info(f"Model loaded from {filepath}")
        else:
            # Assume it's just the state dict
            self.load_state_dict(checkpoint, strict=strict)
            logger.info(f"Model state dict loaded from {filepath}")
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")
    
    def get_layer_outputs(self, x: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """Get outputs from specific layers."""
        outputs = {}
        hooks = []
        
        def hook_fn(name) -> Any:
            def hook(module, input, output) -> Any:
                outputs[name] = output.detach()
            return hook
        
        # Register hooks
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by layer type."""
        param_counts = {}
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                module_type = type(module).__name__
                if module_type not in param_counts:
                    param_counts[module_type] = 0
                
                param_counts[module_type] += sum(p.numel() for p in module.parameters())
        
        return param_counts
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def summary(self) -> str:
        """Get model summary."""
        model_info = self.get_model_info()
        param_counts = self.count_parameters()
        model_size = self.get_model_size_mb()
        
        summary = f"""
Model Summary: {model_info['model_name']}
{'='*50}
Model Type: {model_info['model_type']}
Total Parameters: {model_info['total_parameters']:,}
Trainable Parameters: {model_info['trainable_parameters']:,}
Model Size: {model_size:.2f} MB
Device: {model_info['device']}

Parameters by Layer Type:
{'-'*30}
"""
        
        for layer_type, count in param_counts.items():
            summary += f"{layer_type}: {count:,}\n"
        
        summary += f"\nConfiguration:\n{'-'*15}\n"
        for key, value in model_info['config'].items():
            summary += f"{key}: {value}\n"
        
        return summary


class SimpleMLP(BaseModel):
    """Simple Multi-Layer Perceptron implementation."""
    
    def _build_model(self) -> Any:
        """Build MLP architecture."""
        layers = []
        input_size = self.config.input_size
        
        # Hidden layers
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            elif self.config.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(self.config.dropout_rate))
            
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, self.config.output_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)


# Example usage
if __name__ == "__main__":
    # Create model configuration
    config = ModelConfig(
        model_name="test_mlp",
        model_type="mlp",
        input_size=784,
        output_size=10,
        hidden_sizes=[512, 256, 128],
        activation="relu",
        dropout_rate=0.2,
        batch_norm=True
    )
    
    # Create model
    model = SimpleMLP(config)
    
    # Print model summary
    print(model.summary())
    
    # Test forward pass
    x = torch.randn(32, 784)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Save model
    model.save_model("checkpoints/test_model.pth")
    
    # Load model
    new_model = SimpleMLP(config)
    new_model.load_model("checkpoints/test_model.pth")
    
    print("Model saved and loaded successfully!") 
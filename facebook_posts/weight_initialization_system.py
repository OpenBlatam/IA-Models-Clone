"""
🔧 Weight Initialization and Normalization System
================================================

Comprehensive weight initialization and normalization techniques for deep learning models.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
import logging
import json
import matplotlib.pyplot as plt

# Import experiment tracking if available
try:
    from .experiment_tracking import ExperimentTracker
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError:
    EXPERIMENT_TRACKING_AVAILABLE = False
    ExperimentTracker = None

logger = logging.getLogger(__name__)


@dataclass
class WeightInitConfig:
    """Configuration for weight initialization strategies"""
    
    # General settings
    method: str = "xavier_uniform"
    gain: float = 1.0
    fan_mode: str = "fan_in"
    nonlinearity: str = "leaky_relu"
    
    # Layer-specific settings
    conv_init: str = "kaiming_uniform"
    linear_init: str = "xavier_uniform"
    lstm_init: str = "orthogonal"
    attention_init: str = "xavier_uniform"
    
    # Normalization settings
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    use_group_norm: bool = False
    
    # BatchNorm parameters
    batch_norm_momentum: float = 0.1
    batch_norm_eps: float = 1e-5
    
    # Monitoring
    track_initialization: bool = True
    save_initialization_stats: bool = True


@dataclass
class InitializationStats:
    """Statistics about weight initialization"""
    
    layer_name: str
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    weight_norm: float
    bias_mean: Optional[float] = None
    bias_std: Optional[float] = None
    fan_in: Optional[int] = None
    fan_out: Optional[int] = None
    initialization_method: str = ""
    activation_function: str = ""


class WeightInitializer:
    """Comprehensive weight initialization and normalization system."""
    
    def __init__(self, config: WeightInitConfig, experiment_tracker: Optional[ExperimentTracker] = None):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.stats: List[InitializationStats] = []
        self.layer_stats: Dict[str, InitializationStats] = {}
        
        self.logger = logging.getLogger(f"{__name__}.WeightInitializer")
        
        # Available initialization methods
        self.init_methods = {
            'xavier_uniform': self._xavier_uniform_init,
            'xavier_normal': self._xavier_normal_init,
            'kaiming_uniform': self._kaiming_uniform_init,
            'kaiming_normal': self._kaiming_normal_init,
            'orthogonal': self._orthogonal_init,
            'sparse': self._sparse_init,
            'zeros': self._zeros_init,
            'ones': self._ones_init,
            'constant': self._constant_init,
            'uniform': self._uniform_init,
            'normal': self._normal_init
        }
        
        self.logger.info(f"WeightInitializer initialized with method: {config.method}")
    
    def initialize_model(self, model: nn.Module, track_stats: bool = True) -> Dict[str, Any]:
        """Initialize all weights in a PyTorch model."""
        
        self.logger.info(f"Initializing model: {model.__class__.__name__}")
        
        if track_stats:
            self.stats.clear()
            self.layer_stats.clear()
        
        total_params = 0
        initialized_layers = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                self._initialize_layer(module, name, track_stats)
                initialized_layers += 1
                total_params += module.weight.numel()
                
                if hasattr(module, 'bias') and module.bias is not None:
                    total_params += module.bias.numel()
        
        summary = {
            'total_layers': initialized_layers,
            'total_parameters': total_params,
            'initialization_method': self.config.method
        }
        
        self.logger.info(f"Model initialization complete: {summary}")
        
        if self.experiment_tracker and track_stats:
            self._track_initialization_to_experiment(summary)
        
        return summary
    
    def _initialize_layer(self, module: nn.Module, name: str, track_stats: bool):
        """Initialize weights for a specific layer."""
        
        init_method = self._get_layer_init_method(module)
        
        if hasattr(module, 'weight') and module.weight is not None:
            self._apply_initialization(module.weight, init_method, module, name)
        
        if hasattr(module, 'bias') and module.bias is not None:
            self._initialize_bias(module.bias, module, name)
        
        if track_stats:
            self._track_layer_stats(module, name, init_method)
    
    def _get_layer_init_method(self, module: nn.Module) -> str:
        """Get the appropriate initialization method for a layer type."""
        
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return self.config.conv_init
        elif isinstance(module, nn.Linear):
            return self.config.linear_init
        elif isinstance(module, (nn.LSTM, nn.LSTMCell)):
            return self.config.lstm_init
        elif 'attention' in module.__class__.__name__.lower():
            return self.config.attention_init
        else:
            return self.config.method
    
    def _apply_initialization(self, weight: nn.Parameter, method: str, module: nn.Module, name: str):
        """Apply the specified initialization method to weights."""
        
        if method in self.init_methods:
            self.init_methods[method](weight, module, name)
        else:
            self.logger.warning(f"Unknown initialization method: {method}, using default")
            self._xavier_uniform_init(weight, module, name)
    
    def _xavier_uniform_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Xavier/Glorot uniform initialization."""
        fan_in, fan_out = self._calculate_fan_in_fan_out(module)
        bound = self.config.gain * np.sqrt(6.0 / (fan_in + fan_out))
        init.uniform_(weight, -bound, bound)
    
    def _xavier_normal_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Xavier/Glorot normal initialization."""
        fan_in, fan_out = self._calculate_fan_in_fan_out(module)
        std = self.config.gain * np.sqrt(2.0 / (fan_in + fan_out))
        init.normal_(weight, 0, std)
    
    def _kaiming_uniform_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Kaiming/He uniform initialization."""
        fan_in, fan_out = self._calculate_fan_in_fan_out(module)
        fan = fan_in if self.config.fan_mode == "fan_in" else fan_out
        if self.config.fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        
        bound = self.config.gain * np.sqrt(3.0 / fan)
        init.uniform_(weight, -bound, bound)
    
    def _kaiming_normal_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Kaiming/He normal initialization."""
        fan_in, fan_out = self._calculate_fan_in_fan_out(module)
        fan = fan_in if self.config.fan_mode == "fan_in" else fan_out
        if self.config.fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        
        std = self.config.gain / np.sqrt(fan)
        init.normal_(weight, 0, std)
    
    def _orthogonal_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Orthogonal initialization."""
        init.orthogonal_(weight, gain=self.config.gain)
    
    def _sparse_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Sparse initialization."""
        init.sparse_(weight, sparsity=0.1, std=0.01)
    
    def _zeros_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Initialize weights to zero."""
        init.zeros_(weight)
    
    def _ones_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Initialize weights to one."""
        init.ones_(weight)
    
    def _constant_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Initialize weights to a constant value."""
        init.constant_(weight, 0.0)
    
    def _uniform_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Uniform initialization."""
        init.uniform_(weight, -1.0, 1.0)
    
    def _normal_init(self, weight: nn.Parameter, module: nn.Module, name: str):
        """Normal initialization."""
        init.normal_(weight, 0, 1.0)
    
    def _initialize_bias(self, bias: nn.Parameter, module: nn.Module, name: str):
        """Initialize bias parameters."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in, _ = self._calculate_fan_in_fan_out(module)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
        else:
            init.zeros_(bias)
    
    def _calculate_fan_in_fan_out(self, module: nn.Module) -> Tuple[int, int]:
        """Calculate fan-in and fan-out for a module."""
        
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            fan_out = module.out_features
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if isinstance(module, nn.Conv1d):
                kernel_size = module.kernel_size[0]
            elif isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[0] * module.kernel_size[1]
            else:  # Conv3d
                kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
            
            fan_in = module.in_channels * kernel_size
            fan_out = module.out_channels * kernel_size
        elif isinstance(module, (nn.LSTM, nn.LSTMCell)):
            fan_in = module.input_size
            fan_out = module.hidden_size
        else:
            fan_in = fan_out = 1
        
        return fan_in, fan_out
    
    def _track_layer_stats(self, module: nn.Module, name: str, init_method: str):
        """Track initialization statistics for a layer."""
        
        weight = module.weight
        bias = getattr(module, 'bias', None)
        
        weight_mean = weight.mean().item()
        weight_std = weight.std().item()
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        weight_norm = weight.norm().item()
        
        bias_mean = bias.mean().item() if bias is not None else None
        bias_std = bias.std().item() if bias is not None else None
        
        fan_in, fan_out = self._calculate_fan_in_fan_out(module)
        activation = self._get_activation_function(module)
        
        stats = InitializationStats(
            layer_name=name,
            weight_mean=weight_mean,
            weight_std=weight_std,
            weight_min=weight_min,
            weight_max=weight_max,
            weight_norm=weight_norm,
            bias_mean=bias_mean,
            bias_std=bias_std,
            fan_in=fan_in,
            fan_out=fan_out,
            initialization_method=init_method,
            activation_function=activation
        )
        
        self.stats.append(stats)
        self.layer_stats[name] = stats
    
    def _get_activation_function(self, module: nn.Module) -> str:
        """Determine the activation function used in a module."""
        
        if hasattr(module, 'activation'):
            return str(module.activation)
        elif hasattr(module, 'act'):
            return str(module.act)
        elif hasattr(module, 'nonlinearity'):
            return str(module.nonlinearity)
        else:
            return "unknown"
    
    def get_initialization_summary(self) -> Dict[str, Any]:
        """Get a summary of initialization statistics."""
        
        if not self.stats:
            return {"error": "No initialization statistics available"}
        
        summary = {
            'total_layers': len(self.stats),
            'initialization_methods': list(set(stat.initialization_method for stat in self.stats)),
            'activation_functions': list(set(stat.activation_function for stat in self.stats)),
            'weight_statistics': {
                'mean': np.mean([stat.weight_mean for stat in self.stats]),
                'std': np.mean([stat.weight_std for stat in self.stats]),
                'min': np.min([stat.weight_min for stat in self.stats]),
                'max': np.max([stat.weight_max for stat in self.stats]),
                'norm': np.mean([stat.weight_norm for stat in self.stats])
            },
            'layer_details': [
                {
                    'name': stat.layer_name,
                    'method': stat.initialization_method,
                    'activation': stat.activation_function,
                    'weight_mean': stat.weight_mean,
                    'weight_std': stat.weight_std,
                    'weight_norm': stat.weight_norm,
                    'fan_in': stat.fan_in,
                    'fan_out': stat.fan_out
                }
                for stat in self.stats
            ]
        }
        
        return summary
    
    def save_initialization_stats(self, filepath: str) -> None:
        """Save initialization statistics to a JSON file."""
        
        if not self.stats:
            self.logger.warning("No statistics available to save")
            return
        
        serializable_stats = []
        for stat in self.stats:
            serializable_stats.append({
                'layer_name': stat.layer_name,
                'weight_mean': stat.weight_mean,
                'weight_std': stat.weight_std,
                'weight_min': stat.weight_min,
                'weight_max': stat.weight_max,
                'weight_norm': stat.weight_norm,
                'bias_mean': stat.bias_mean,
                'bias_std': stat.bias_std,
                'fan_in': stat.fan_in,
                'fan_out': stat.fan_out,
                'initialization_method': stat.initialization_method,
                'activation_function': stat.activation_function
            })
        
        data = {
            'config': self.config.__dict__,
            'summary': self.get_initialization_summary(),
            'layer_stats': serializable_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Initialization statistics saved to: {filepath}")
    
    def _track_initialization_to_experiment(self, summary: Dict[str, Any]) -> None:
        """Track initialization information to the experiment tracker."""
        
        if not self.experiment_tracker:
            return
        
        try:
            self.experiment_tracker.log_metric('initialization/total_layers', summary['total_layers'])
            self.experiment_tracker.log_metric('initialization/total_parameters', summary['total_parameters'])
            
            if self.stats:
                weight_means = [stat.weight_mean for stat in self.stats]
                weight_stds = [stat.weight_std for stat in self.stats]
                weight_norms = [stat.weight_norm for stat in self.stats]
                
                self.experiment_tracker.log_metric('initialization/weight_mean', np.mean(weight_means))
                self.experiment_tracker.log_metric('initialization/weight_std', np.mean(weight_stds))
                self.experiment_tracker.log_metric('initialization/weight_norm', np.mean(weight_norms))
                self.experiment_tracker.log_metric('initialization/weight_min', np.min([stat.weight_min for stat in self.stats]))
                self.experiment_tracker.log_metric('initialization/weight_max', np.max([stat.weight_max for stat in self.stats]))
            
            self.experiment_tracker.log_config({
                'weight_initialization': self.config.__dict__
            })
            
            self.logger.info("Initialization information tracked to experiment tracker")
            
        except Exception as e:
            self.logger.error(f"Failed to track initialization to experiment tracker: {e}")


def create_weight_initializer(
    method: str = "xavier_uniform",
    gain: float = 1.0,
    fan_mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    **kwargs
) -> WeightInitializer:
    """Factory function to create a WeightInitializer with common configurations."""
    
    config = WeightInitConfig(
        method=method,
        gain=gain,
        fan_mode=fan_mode,
        nonlinearity=nonlinearity,
        **kwargs
    )
    
    return WeightInitializer(config)


def get_initialization_recommendations(architecture: str) -> Dict[str, Any]:
    """Get recommended initialization strategies for common architectures."""
    
    recommendations = {
        'cnn': {
            'conv_init': 'kaiming_uniform',
            'linear_init': 'xavier_uniform',
            'nonlinearity': 'relu',
            'use_batch_norm': True,
            'use_layer_norm': False
        },
        'rnn': {
            'conv_init': 'xavier_uniform',
            'linear_init': 'orthogonal',
            'lstm_init': 'orthogonal',
            'nonlinearity': 'tanh',
            'use_batch_norm': False,
            'use_layer_norm': True
        },
        'transformer': {
            'conv_init': 'xavier_uniform',
            'linear_init': 'xavier_uniform',
            'attention_init': 'xavier_uniform',
            'nonlinearity': 'relu',
            'use_batch_norm': False,
            'use_layer_norm': True
        },
        'mlp': {
            'conv_init': 'xavier_uniform',
            'linear_init': 'xavier_uniform',
            'nonlinearity': 'relu',
            'use_batch_norm': True,
            'use_layer_norm': False
        }
    }
    
    return recommendations.get(architecture.lower(), recommendations['mlp'])


if __name__ == "__main__":
    # Example usage
    print("🔧 Weight Initialization and Normalization System")
    print("=" * 50)
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Create model and initializer
    model = SimpleModel()
    
    # Get recommendations for CNN architecture
    recommendations = get_initialization_recommendations('cnn')
    print(f"CNN Recommendations: {recommendations}")
    
    # Create initializer with recommendations
    config = WeightInitConfig(**recommendations)
    initializer = WeightInitializer(config)
    
    # Initialize the model
    summary = initializer.initialize_model(model)
    print(f"Initialization Summary: {summary}")
    
    # Get detailed statistics
    stats = initializer.get_initialization_summary()
    print(f"Layer Statistics: {len(stats['layer_details'])} layers initialized")
    
    # Save statistics
    initializer.save_initialization_stats("initialization_stats.json")
    
    print("\n✅ Weight initialization system demonstration completed!")

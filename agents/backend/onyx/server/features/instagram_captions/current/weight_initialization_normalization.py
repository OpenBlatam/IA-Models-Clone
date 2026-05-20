"""
Weight Initialization and Normalization System
Comprehensive implementation of proper weight initialization and normalization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class WeightConfig:
    """Configuration for weight initialization and normalization."""
    
    # Model dimensions
    input_size: int = 784
    hidden_size: int = 256
    output_size: int = 10
    num_layers: int = 3
    
    # Initialization parameters
    init_method: str = "xavier_uniform"  # xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal
    init_gain: float = 1.0
    init_std: float = 0.02
    init_range: float = 0.1
    
    # Normalization parameters
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    use_group_norm: bool = False
    use_instance_norm: bool = False
    norm_momentum: float = 0.1
    norm_eps: float = 1e-5
    norm_affine: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1


class WeightInitializer:
    """Comprehensive weight initializer with various techniques."""
    
    @staticmethod
    def xavier_uniform(tensor: torch.Tensor, gain: float = 1.0):
        """Xavier uniform initialization for linear layers."""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        init.uniform_(tensor, -bound, bound)
    
    @staticmethod
    def xavier_normal(tensor: torch.Tensor, gain: float = 1.0):
        """Xavier normal initialization for linear layers."""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        init.normal_(tensor, 0, std)
    
    @staticmethod
    def kaiming_uniform(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
        """Kaiming uniform initialization for convolutional layers."""
        init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def kaiming_normal(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
        """Kaiming normal initialization for convolutional layers."""
        init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def orthogonal(tensor: torch.Tensor, gain: float = 1.0):
        """Orthogonal initialization for recurrent layers."""
        init.orthogonal_(tensor, gain=gain)
    
    @staticmethod
    def sparse(tensor: torch.Tensor, sparsity: float = 0.1, std: float = 0.01):
        """Sparse initialization for sparse layers."""
        init.sparse_(tensor, sparsity=sparsity, std=std)
    
    @staticmethod
    def delta_orthogonal(tensor: torch.Tensor, gain: float = 1.0):
        """Delta orthogonal initialization for recurrent layers."""
        init.delta_orthogonal_(tensor, gain=gain)
    
    @staticmethod
    def calculate_fan_in_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
        """Calculate fan_in and fan_out for a tensor."""
        if tensor.dim() < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        
        if tensor.dim() == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:  # Convolutional
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        
        return fan_in, fan_out
    
    @staticmethod
    def get_optimal_gain(nonlinearity: str) -> float:
        """Get optimal gain for different activation functions."""
        gains = {
            'linear': 1.0,
            'conv1d': 1.0,
            'conv2d': 1.0,
            'conv3d': 1.0,
            'conv_transpose1d': 1.0,
            'conv_transpose2d': 1.0,
            'conv_transpose3d': 1.0,
            'sigmoid': 1.0,
            'tanh': 5.0 / 3,
            'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + 0.01 ** 2)),
            'selu': 3.0 / 4,
            'glu': 1.0,
            'swish': 1.0,
            'gelu': 1.0
        }
        return gains.get(nonlinearity, 1.0)
    
    @staticmethod
    def initialize_weights(module: nn.Module, method: str = "xavier_uniform", **kwargs):
        """Initialize weights for an entire module."""
        for name, param in module.named_parameters():
            if 'weight' in name:
                if method == "xavier_uniform":
                    WeightInitializer.xavier_uniform(param, kwargs.get('gain', 1.0))
                elif method == "xavier_normal":
                    WeightInitializer.xavier_normal(param, kwargs.get('gain', 1.0))
                elif method == "kaiming_uniform":
                    WeightInitializer.kaiming_uniform(param, kwargs.get('mode', 'fan_in'), 
                                                   kwargs.get('nonlinearity', 'leaky_relu'))
                elif method == "kaiming_normal":
                    WeightInitializer.kaiming_normal(param, kwargs.get('mode', 'fan_in'), 
                                                  kwargs.get('nonlinearity', 'leaky_relu'))
                elif method == "orthogonal":
                    WeightInitializer.orthogonal(param, kwargs.get('gain', 1.0))
                elif method == "sparse":
                    WeightInitializer.sparse(param, kwargs.get('sparsity', 0.1), 
                                           kwargs.get('std', 0.01))
                elif method == "delta_orthogonal":
                    WeightInitializer.delta_orthogonal(param, kwargs.get('gain', 1.0))
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
            
            elif 'bias' in name:
                init.zeros_(param)


class NormalizationLayer(nn.Module):
    """Comprehensive normalization layer with multiple techniques."""
    
    def __init__(self, config: WeightConfig, num_features: int, layer_type: str = "batch"):
        super().__init__()
        self.config = config
        self.num_features = num_features
        self.layer_type = layer_type
        
        # Initialize normalization layer
        self.norm_layer = self._create_norm_layer()
    
    def _create_norm_layer(self) -> nn.Module:
        """Create the appropriate normalization layer."""
        if self.layer_type == "batch" and self.config.use_batch_norm:
            return nn.BatchNorm1d(
                self.num_features,
                momentum=self.config.norm_momentum,
                eps=self.config.norm_eps,
                affine=self.config.norm_affine
            )
        elif self.layer_type == "layer" and self.config.use_layer_norm:
            return nn.LayerNorm(
                self.num_features,
                eps=self.config.norm_eps,
                elementwise_affine=self.config.norm_affine
            )
        elif self.layer_type == "group" and self.config.use_group_norm:
            num_groups = min(32, self.num_features // 4)  # Ensure groups divide features
            return nn.GroupNorm(
                num_groups,
                self.num_features,
                eps=self.config.norm_eps,
                affine=self.config.norm_affine
            )
        elif self.layer_type == "instance" and self.config.use_instance_norm:
            return nn.InstanceNorm1d(
                self.num_features,
                momentum=self.config.norm_momentum,
                eps=self.config.norm_eps,
                affine=self.config.norm_affine
            )
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalization layer."""
        return self.norm_layer(x)


class WeightNormalizedLinear(nn.Module):
    """Weight-normalized linear layer implementation."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Weight and bias parameters
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_g = nn.Parameter(torch.ones(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weight-normalized parameters."""
        init.xavier_uniform_(self.weight_v)
        init.ones_(self.weight_g)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight normalization."""
        # Normalize weights
        weight_norm = self.weight_v.norm(dim=1, keepdim=True)
        weight_normalized = self.weight_v / (weight_norm + self.eps)
        
        # Scale by learned parameter
        weight_scaled = weight_normalized * self.weight_g
        
        # Linear transformation
        return F.linear(x, weight_scaled, self.bias)


class SpectralNormalizedLinear(nn.Module):
    """Spectral-normalized linear layer implementation."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize spectral-normalized parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def _spectral_norm(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply spectral normalization to weights."""
        # Power iteration method
        u = torch.randn(weight.size(0), 1, device=weight.device)
        v = torch.randn(weight.size(1), 1, device=weight.device)
        
        for _ in range(1):  # Single iteration for efficiency
            v = F.normalize(torch.mv(weight.t(), u.squeeze()), dim=0, eps=self.eps).unsqueeze(1)
            u = F.normalize(torch.mv(weight, v.squeeze()), dim=0, eps=self.eps).unsqueeze(1)
        
        # Compute spectral norm
        sigma = u.t() @ weight @ v
        
        # Normalize weights
        weight_normalized = weight / sigma
        
        return weight_normalized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral normalization."""
        # Apply spectral normalization
        weight_normalized = self._spectral_norm(self.weight)
        
        # Linear transformation
        return F.linear(x, weight_normalized, self.bias)


class ProperlyInitializedModel(nn.Module):
    """Model demonstrating proper weight initialization and normalization."""
    
    def __init__(self, config: WeightConfig):
        super().__init__()
        self.config = config
        
        # Build layers with proper initialization
        self.layers = nn.ModuleList()
        
        # Input layer
        input_layer = nn.Linear(config.input_size, config.hidden_size)
        WeightInitializer.initialize_weights(input_layer, config.init_method, 
                                          gain=WeightInitializer.get_optimal_gain('relu'))
        self.layers.append(input_layer)
        
        # Hidden layers
        for i in range(config.num_layers - 1):
            # Linear layer
            hidden_layer = nn.Linear(config.hidden_size, config.hidden_size)
            WeightInitializer.initialize_weights(hidden_layer, config.init_method,
                                              gain=WeightInitializer.get_optimal_gain('relu'))
            self.layers.append(hidden_layer)
            
            # Normalization layer
            norm_layer = NormalizationLayer(config, config.hidden_size, "batch")
            self.layers.append(norm_layer)
            
            # Dropout
            if config.dropout_rate > 0:
                self.layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        output_layer = nn.Linear(config.hidden_size, config.output_size)
        WeightInitializer.initialize_weights(output_layer, config.init_method,
                                          gain=WeightInitializer.get_optimal_gain('linear'))
        self.layers.append(output_layer)
        
        # Weight-normalized and spectral-normalized alternatives
        self.weight_norm_layer = WeightNormalizedLinear(config.hidden_size, config.hidden_size)
        self.spectral_norm_layer = SpectralNormalizedLinear(config.hidden_size, config.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 1:  # Not the output layer
                    x = F.relu(x)
            else:
                x = layer(x)
        
        return x
    
    def forward_with_advanced_norm(self, x: torch.Tensor, norm_type: str = "standard") -> torch.Tensor:
        """Forward pass with advanced normalization techniques."""
        if norm_type == "weight_norm":
            x = self.weight_norm_layer(x)
        elif norm_type == "spectral_norm":
            x = self.spectral_norm_layer(x)
        else:
            x = self.forward(x)
        
        return x


class WeightAnalyzer:
    """Analyze weight distributions and properties."""
    
    def __init__(self):
        self.weight_stats = {}
        self.gradient_stats = {}
    
    def analyze_weights(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Analyze weight statistics for all parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data = param.data
                
                self.weight_stats[name] = {
                    'mean': weight_data.mean().item(),
                    'std': weight_data.std().item(),
                    'min': weight_data.min().item(),
                    'max': weight_data.max().item(),
                    'norm': weight_data.norm().item(),
                    'sparsity': (weight_data == 0).float().mean().item()
                }
        
        return self.weight_stats
    
    def analyze_gradients(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Analyze gradient statistics for all parameters."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                
                self.gradient_stats[name] = {
                    'mean': grad_data.mean().item(),
                    'std': grad_data.std().item(),
                    'min': grad_data.min().item(),
                    'max': grad_data.max().item(),
                    'norm': grad_data.norm().item(),
                    'gradient_scale': grad_data.norm() / (param.data.norm() + 1e-8)
                }
        
        return self.gradient_stats
    
    def plot_weight_distributions(self, model: nn.Module):
        """Plot weight distributions for all layers."""
        self.analyze_weights(model)
        
        num_layers = len(self.weight_stats)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(15, 8))
        if num_layers == 1:
            axes = [axes]
        elif num_layers <= 2:
            axes = axes.flatten()
        
        for i, (name, stats) in enumerate(self.weight_stats.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Get weights for this layer
                for param_name, param in model.named_parameters():
                    if param_name == name:
                        weights = param.data.flatten().cpu().numpy()
                        break
                
                # Plot histogram
                ax.hist(weights, bins=50, alpha=0.7, density=True)
                ax.set_title(f'{name}\nμ={stats["mean"]:.3f}, σ={stats["std"]:.3f}')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distributions(self, model: nn.Module):
        """Plot gradient distributions for all layers."""
        self.analyze_gradients(model)
        
        if not self.gradient_stats:
            print("No gradients available for analysis")
            return
        
        num_layers = len(self.gradient_stats)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(15, 8))
        if num_layers == 1:
            axes = [axes]
        elif num_layers <= 2:
            axes = axes.flatten()
        
        for i, (name, stats) in enumerate(self.gradient_stats.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Get gradients for this layer
                for param_name, param in model.named_parameters():
                    if param_name == name:
                        gradients = param.grad.data.flatten().cpu().numpy()
                        break
                
                # Plot histogram
                ax.hist(gradients, bins=50, alpha=0.7, density=True)
                ax.set_title(f'{name}\nμ={stats["mean"]:.3f}, σ={stats["std"]:.3f}')
                ax.set_xlabel('Gradient Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class WeightInitializationTester:
    """Test different weight initialization methods."""
    
    def __init__(self, config: WeightConfig):
        self.config = config
        self.results = {}
    
    def test_initialization_methods(self) -> Dict[str, Dict[str, float]]:
        """Test various initialization methods and analyze results."""
        methods = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"]
        
        for method in methods:
            # Create model with specific initialization
            model = ProperlyInitializedModel(self.config)
            
            # Reinitialize weights with the method
            WeightInitializer.initialize_weights(model, method)
            
            # Analyze weights
            analyzer = WeightAnalyzer()
            weight_stats = analyzer.analyze_weights(model)
            
            # Calculate summary statistics
            all_means = [stats['mean'] for stats in weight_stats.values()]
            all_stds = [stats['std'] for stats in weight_stats.values()]
            all_norms = [stats['norm'] for stats in weight_stats.values()]
            
            self.results[method] = {
                'mean_weight_mean': np.mean(all_means),
                'std_weight_mean': np.std(all_means),
                'mean_weight_std': np.mean(all_stds),
                'std_weight_std': np.std(all_stds),
                'mean_weight_norm': np.mean(all_norms),
                'std_weight_norm': np.std(all_norms)
            }
        
        return self.results
    
    def plot_initialization_comparison(self):
        """Plot comparison of different initialization methods."""
        if not self.results:
            self.test_initialization_methods()
        
        methods = list(self.results.keys())
        metrics = ['mean_weight_mean', 'mean_weight_std', 'mean_weight_norm']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [self.results[method][metric] for method in methods]
            axes[i].bar(methods, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = WeightConfig(
        input_size=784,
        hidden_size=256,
        output_size=10,
        num_layers=3,
        init_method="xavier_uniform",
        use_batch_norm=True,
        use_layer_norm=True,
        dropout_rate=0.1
    )
    
    # Initialize model with proper weight initialization
    model = ProperlyInitializedModel(config)
    logging.info("Model initialized with proper weight initialization")
    
    # Analyze weights
    analyzer = WeightAnalyzer()
    weight_stats = analyzer.analyze_weights(model)
    
    logging.info("Weight Statistics:")
    for name, stats in weight_stats.items():
        logging.info(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                    f"norm={stats['norm']:.4f}")
    
    # Test different initialization methods
    tester = WeightInitializationTester(config)
    results = tester.test_initialization_methods()
    
    logging.info("Initialization Method Comparison:")
    for method, metrics in results.items():
        logging.info(f"{method}: {metrics}")
    
    # Create sample input
    batch_size = 32
    sample_input = torch.randn(batch_size, config.input_size)
    
    # Forward pass
    output = model(sample_input)
    logging.info(f"Output shape: {output.shape}")
    
    # Test advanced normalization
    output_weight_norm = model.forward_with_advanced_norm(sample_input, "weight_norm")
    output_spectral_norm = model.forward_with_advanced_norm(sample_input, "spectral_norm")
    
    logging.info(f"Weight norm output shape: {output_weight_norm.shape}")
    logging.info(f"Spectral norm output shape: {output_spectral_norm.shape}")
    
    # Visualize results
    analyzer.plot_weight_distributions(model)
    tester.plot_initialization_comparison()
    
    logging.info("Weight initialization and normalization demonstration completed successfully!")






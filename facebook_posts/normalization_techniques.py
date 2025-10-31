from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
Advanced Normalization Techniques
Comprehensive implementation of modern normalization methods and best practices.
"""



class NormalizationType(Enum):
    """Types of normalization techniques."""
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"
    WEIGHT_NORM = "weight_norm"
    SPECTRAL_NORM = "spectral_norm"
    ADAPTIVE_NORM = "adaptive_norm"
    SWITCHABLE_NORM = "switchable_norm"
    CROSS_NORM = "cross_norm"


@dataclass
class NormalizationConfig:
    """Configuration for normalization layers."""
    normalization_type: NormalizationType = NormalizationType.LAYER_NORM
    num_features: int = 512
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    num_groups: int = 32
    elementwise_affine: bool = True
    power_iterations: int = 1
    adaptive_momentum: float = 0.01


class AdvancedLayerNorm(nn.Module):
    """Advanced layer normalization with additional features."""
    
    def __init__(self, normalized_shape: Union[int, List[int]], 
                 eps: float = 1e-5, elementwise_affine: bool = True,
                 bias_correction: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias_correction = bias_correction
        
        if elementwise_affine:
            if isinstance(normalized_shape, int):
                normalized_shape = [normalized_shape]
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with advanced layer normalization."""
        # Calculate mean and variance
        if isinstance(self.normalized_shape, int):
            dims = list(range(-self.normalized_shape, 0))
        else:
            dims = list(range(-len(self.normalized_shape), 0))
        
        mean = input_tensor.mean(dim=dims, keepdim=True)
        var = input_tensor.var(dim=dims, keepdim=True, unbiased=False)
        
        # Apply bias correction if enabled
        if self.bias_correction and self.training:
            # Use running statistics for bias correction
            if not hasattr(self, 'running_mean'):
                self.register_buffer('running_mean', torch.zeros_like(mean))
                self.register_buffer('running_var', torch.ones_like(var))
            
            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean.detach()
            self.running_var = 0.9 * self.running_var + 0.1 * var.detach()
            
            # Use running statistics for normalization
            normalized = (input_tensor - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            normalized = (input_tensor - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
        
        return normalized


class AdaptiveBatchNorm(nn.Module):
    """Adaptive batch normalization that adjusts to data distribution."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 adaptive_momentum: float = 0.01):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.adaptive_momentum = adaptive_momentum
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive batch normalization."""
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        
        # Calculate batch statistics
        if self.training:
            mean = input_tensor.mean(dim=0)
            var = input_tensor.var(dim=0, unbiased=False)
            
            # Adaptive momentum based on batch statistics
            if self.track_running_stats:
                # Adjust momentum based on batch size and data distribution
                batch_size = input_tensor.size(0)
                adaptive_factor = min(1.0, batch_size / 32.0)  # Normalize by typical batch size
                
                # Update running statistics with adaptive momentum
                self.running_mean = (1 - exponential_average_factor * adaptive_factor) * self.running_mean + \
                                  exponential_average_factor * adaptive_factor * mean.detach()
                self.running_var = (1 - exponential_average_factor * adaptive_factor) * self.running_var + \
                                 exponential_average_factor * adaptive_factor * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        normalized = (input_tensor - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.affine:
            normalized = normalized * self.weight + self.bias
        
        return normalized


class SwitchableNormalization(nn.Module):
    """Switchable normalization that can switch between different normalization methods."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, num_groups: int = 32):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_groups = num_groups
        
        # Create different normalization layers
        self.batch_norm = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.layer_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
        self.instance_norm = nn.InstanceNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.group_norm = nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)
        
        # Switchable weights
        self.switch_weights = nn.Parameter(torch.ones(4))  # 4 normalization methods
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with switchable normalization."""
        # Apply each normalization method
        batch_norm_out = self.batch_norm(input_tensor)
        layer_norm_out = self.layer_norm(input_tensor)
        instance_norm_out = self.instance_norm(input_tensor)
        group_norm_out = self.group_norm(input_tensor)
        
        # Weighted combination
        weights = F.softmax(self.switch_weights, dim=0)
        
        output = (weights[0] * batch_norm_out + 
                 weights[1] * layer_norm_out + 
                 weights[2] * instance_norm_out + 
                 weights[3] * group_norm_out)
        
        return output


class CrossNormalization(nn.Module):
    """Cross normalization that normalizes across different dimensions."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with cross normalization."""
        # Normalize across batch and feature dimensions
        batch_size = input_tensor.size(0)
        
        # Reshape for cross normalization
        reshaped = input_tensor.view(batch_size, -1)
        
        # Calculate statistics across all dimensions except batch
        mean = reshaped.mean(dim=1, keepdim=True)
        var = reshaped.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        output = normalized.view_as(input_tensor)
        
        # Apply affine transformation
        if self.affine:
            output = output * self.weight + self.bias
        
        return output


class WeightNormalization(nn.Module):
    """Weight normalization for linear layers."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight and bias
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_g = nn.Parameter(torch.ones(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weight normalization parameters."""
        nn.init.xavier_uniform_(self.weight_v)
        nn.init.ones_(self.weight_g)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight normalization."""
        # Normalize weight
        weight_norm = self.weight_v / (self.weight_v.norm(p=2, dim=1, keepdim=True) + 1e-8)
        weight = weight_norm * self.weight_g
        
        # Apply linear transformation
        output = F.linear(input_tensor, weight, self.bias)
        
        return output


class SpectralNormalization(nn.Module):
    """Spectral normalization for convolutional and linear layers."""
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        # Register buffer for u vector
        self.register_buffer('u', None)
        self._make_params()
    
    def _make_params(self) -> Any:
        """Make parameters for spectral normalization."""
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1))
        u = F.normalize(u, dim=0, eps=1e-12)
        self.register_buffer('u', u)
    
    def _power_method(self, w, eps=1e-12) -> Any:
        """Power method for spectral normalization."""
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w.view(height, -1).t(), self.u), dim=0, eps=eps)
            u = F.normalize(torch.mv(w.view(height, -1), v), dim=0, eps=eps)
        
        sigma = u.t().mm(w.view(height, -1)).mm(v)
        return u, v, sigma
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral normalization."""
        w = getattr(self.module, self.name)
        u, v, sigma = self._power_method(w)
        
        # Update u buffer
        self.u.copy_(u.detach())
        
        # Normalize weight
        w_sn = w / sigma
        
        # Apply normalized weight
        setattr(self.module, self.name, w_sn)
        output = self.module(input_tensor)
        setattr(self.module, self.name, w)
        
        return output


class NormalizationFactory:
    """Factory for creating different normalization layers."""
    
    @staticmethod
    def create_normalization(norm_type: NormalizationType, 
                           config: NormalizationConfig) -> nn.Module:
        """Create normalization layer based on type and configuration."""
        if norm_type == NormalizationType.BATCH_NORM:
            return nn.BatchNorm1d(
                config.num_features,
                eps=config.eps,
                momentum=config.momentum,
                affine=config.affine,
                track_running_stats=config.track_running_stats
            )
        elif norm_type == NormalizationType.LAYER_NORM:
            return AdvancedLayerNorm(
                config.num_features,
                eps=config.eps,
                elementwise_affine=config.elementwise_affine
            )
        elif norm_type == NormalizationType.INSTANCE_NORM:
            return nn.InstanceNorm1d(
                config.num_features,
                eps=config.eps,
                momentum=config.momentum,
                affine=config.affine,
                track_running_stats=config.track_running_stats
            )
        elif norm_type == NormalizationType.GROUP_NORM:
            return nn.GroupNorm(
                config.num_groups,
                config.num_features,
                eps=config.eps,
                affine=config.affine
            )
        elif norm_type == NormalizationType.WEIGHT_NORM:
            return WeightNormalization(
                config.num_features,
                config.num_features
            )
        elif norm_type == NormalizationType.SPECTRAL_NORM:
            # Create a dummy module for spectral normalization
            dummy_module = nn.Linear(config.num_features, config.num_features)
            return SpectralNormalization(dummy_module)
        elif norm_type == NormalizationType.ADAPTIVE_NORM:
            return AdaptiveBatchNorm(
                config.num_features,
                eps=config.eps,
                momentum=config.momentum,
                affine=config.affine,
                track_running_stats=config.track_running_stats,
                adaptive_momentum=config.adaptive_momentum
            )
        elif norm_type == NormalizationType.SWITCHABLE_NORM:
            return SwitchableNormalization(
                config.num_features,
                eps=config.eps,
                momentum=config.momentum,
                affine=config.affine,
                num_groups=config.num_groups
            )
        elif norm_type == NormalizationType.CROSS_NORM:
            return CrossNormalization(
                config.num_features,
                eps=config.eps,
                affine=config.affine
            )
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")


class NormalizationAnalyzer:
    """Analyze the effects of different normalization methods."""
    
    def __init__(self) -> Any:
        self.analysis_results = {}
    
    def analyze_normalization(self, input_tensor: torch.Tensor,
                            norm_configs: List[NormalizationConfig]) -> Dict[str, Any]:
        """Analyze different normalization methods."""
        results = {}
        
        for config in norm_configs:
            try:
                # Create normalization layer
                norm_layer = NormalizationFactory.create_normalization(
                    config.normalization_type, config
                )
                
                # Apply normalization
                output = norm_layer(input_tensor)
                
                # Analyze output statistics
                output_stats = self._analyze_output_statistics(output)
                
                results[config.normalization_type.value] = {
                    'config': config,
                    'output_stats': output_stats,
                    'success': True
                }
                
            except Exception as e:
                results[config.normalization_type.value] = {
                    'config': config,
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def _analyze_output_statistics(self, output: torch.Tensor) -> Dict[str, float]:
        """Analyze output statistics."""
        return {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item(),
            'norm_l1': output.norm(p=1).item(),
            'norm_l2': output.norm(p=2).item(),
            'norm_inf': output.norm(p=float('inf')).item()
        }


def demonstrate_normalization_techniques():
    """Demonstrate different normalization techniques."""
    print("Normalization Techniques Demonstration")
    print("=" * 50)
    
    # Create input tensor
    batch_size = 4
    seq_length = 10
    num_features = 128
    
    input_tensor = torch.randn(batch_size, seq_length, num_features)
    
    # Create different normalization configurations
    norm_configs = [
        NormalizationConfig(normalization_type=NormalizationType.LAYER_NORM, num_features=num_features),
        NormalizationConfig(normalization_type=NormalizationType.BATCH_NORM, num_features=num_features),
        NormalizationConfig(normalization_type=NormalizationType.GROUP_NORM, num_features=num_features),
        NormalizationConfig(normalization_type=NormalizationType.ADAPTIVE_NORM, num_features=num_features),
        NormalizationConfig(normalization_type=NormalizationType.SWITCHABLE_NORM, num_features=num_features),
        NormalizationConfig(normalization_type=NormalizationType.CROSS_NORM, num_features=num_features)
    ]
    
    # Analyze normalization methods
    analyzer = NormalizationAnalyzer()
    results = analyzer.analyze_normalization(input_tensor, norm_configs)
    
    # Print results
    for method_name, result in results.items():
        print(f"\n{method_name.upper()}:")
        if result['success']:
            stats = result['output_stats']
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  L2 Norm: {stats['norm_l2']:.6f}")
        else:
            print(f"  Error: {result['error']}")
    
    return results


if __name__ == "__main__":
    # Demonstrate normalization techniques
    results = demonstrate_normalization_techniques()
    print("\nNormalization techniques demonstration completed!") 
"""
Weight Initialization and Normalization System

This module provides comprehensive weight initialization and normalization techniques
for deep learning models. It includes:

1. Various weight initialization strategies (Xavier, Kaiming, Orthogonal, etc.)
2. Weight normalization techniques (LayerNorm, BatchNorm, InstanceNorm, etc.)
3. Custom initialization schemes for different architectures
4. Initialization analysis and debugging tools
5. Integration with custom model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings


class WeightInitializer:
    """Comprehensive weight initialization system for neural networks."""
    
    @staticmethod
    def xavier_uniform(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Xavier uniform initialization."""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    
    @staticmethod
    def xavier_normal(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Xavier normal initialization."""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        with torch.no_grad():
            tensor.normal_(0, std)
    
    @staticmethod
    def kaiming_uniform(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', 
                        nonlinearity: str = 'leaky_relu') -> None:
        """Kaiming uniform initialization."""
        init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def kaiming_normal(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', 
                       nonlinearity: str = 'leaky_relu') -> None:
        """Kaiming normal initialization."""
        init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Orthogonal initialization."""
        init.orthogonal_(tensor, gain=gain)
    
    @staticmethod
    def sparse(tensor: torch.Tensor, sparsity: float = 0.1, std: float = 0.01) -> None:
        """Sparse initialization."""
        with torch.no_grad():
            tensor.zero_()
            num_nonzero = int(tensor.numel() * sparsity)
            indices = torch.randperm(tensor.numel())[:num_nonzero]
            tensor.view(-1)[indices] = torch.randn(num_nonzero) * std
    
    @staticmethod
    def lstm_init(tensor: torch.Tensor, num_layers: int = 1) -> None:
        """LSTM-specific initialization."""
        if tensor.ndim != 2:
            raise ValueError("LSTM initialization requires 2D tensor")
        
        WeightInitializer.orthogonal(tensor, gain=1.0)
        with torch.no_grad():
            tensor.mul_(1.0 / math.sqrt(num_layers))
    
    @staticmethod
    def transformer_init(tensor: torch.Tensor, d_model: int) -> None:
        """Transformer-specific initialization."""
        if tensor.ndim != 2:
            raise ValueError("Transformer initialization requires 2D tensor")
        
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        std = math.sqrt(2.0 / (fan_in + fan_out)) / math.sqrt(d_model)
        
        with torch.no_grad():
            tensor.normal_(0, std)
    
    @staticmethod
    def conv_init(tensor: torch.Tensor, mode: str = 'fan_out') -> None:
        """Convolution-specific initialization."""
        if tensor.ndim < 2:
            raise ValueError("Convolution initialization requires at least 2D tensor")
        
        WeightInitializer.kaiming_normal(tensor, mode=mode, nonlinearity='relu')
    
    @staticmethod
    def initialize_model(model: nn.Module, 
                        init_method: str = 'xavier_uniform',
                        **kwargs) -> None:
        """Initialize all parameters in a model."""
        init_func = getattr(WeightInitializer, init_method, None)
        if init_func is None:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    WeightInitializer.conv_init(param)
                elif 'lstm' in name or 'rnn' in name:
                    WeightInitializer.lstm_init(param)
                elif 'transformer' in name or 'attention' in name:
                    WeightInitializer.transformer_init(param)
                else:
                    init_func(param, **kwargs)
            elif 'bias' in name:
                init.zeros_(param)


class WeightNormalizer:
    """Weight normalization techniques for neural networks."""
    
    @staticmethod
    def weight_norm(module: nn.Module, name: str = 'weight', dim: int = 0) -> None:
        """Apply weight normalization to a module."""
        if not hasattr(module, name):
            raise ValueError(f"Module {module} has no parameter {name}")
        
        weight = getattr(module, name)
        if weight is None:
            raise ValueError(f"Parameter {name} is None")
        
        delattr(module, name)
        module.register_parameter(name + '_g', nn.Parameter(weight.norm(dim=dim, keepdim=True)))
        module.register_parameter(name + '_v', nn.Parameter(weight))
        
        module.register_forward_hook(
            lambda mod, inp, outp: setattr(mod, name, 
                F.normalize(getattr(mod, name + '_v'), dim=dim) * getattr(mod, name + '_g'))
        )
    
    @staticmethod
    def apply_normalization(model: nn.Module, 
                           norm_type: str = 'weight_norm',
                           **kwargs) -> None:
        """Apply normalization to all applicable modules in a model."""
        norm_func = getattr(WeightNormalizer, norm_type, None)
        if norm_func is None:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                norm_func(module, **kwargs)


class InitializationAnalyzer:
    """Analyze and debug weight initialization."""
    
    @staticmethod
    def analyze_weights(model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Analyze weight statistics for all parameters in a model."""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data
                stats[name] = {
                    'mean': param_data.mean().item(),
                    'std': param_data.std().item(),
                    'min': param_data.min().item(),
                    'max': param_data.max().item(),
                    'norm': param_data.norm().item(),
                    'norm_l1': param_data.norm(1).item(),
                    'numel': param_data.numel(),
                    'shape': list(param_data.shape)
                }
        
        return stats
    
    @staticmethod
    def check_initialization_quality(model: nn.Module, 
                                   target_std: float = 0.1,
                                   tolerance: float = 0.5) -> Dict[str, Any]:
        """Check the quality of weight initialization."""
        stats = InitializationAnalyzer.analyze_weights(model)
        quality_report = {
            'overall_score': 0.0,
            'parameter_scores': {},
            'warnings': [],
            'recommendations': []
        }
        
        total_score = 0.0
        num_params = len(stats)
        
        for name, param_stats in stats.items():
            param_score = 0.0
            warnings = []
            
            # Check standard deviation
            std = param_stats['std']
            if abs(std - target_std) > target_std * tolerance:
                warnings.append(f"Standard deviation {std:.4f} deviates from target {target_std:.4f}")
                param_score -= 1
            else:
                param_score += 1
            
            # Check for extreme values
            if param_stats['max'] > 10.0 or param_stats['min'] < -10.0:
                warnings.append("Extreme weight values detected")
                param_score -= 1
            
            # Check for zero weights
            if param_stats['norm'] < 1e-6:
                warnings.append("Very small weight norm detected")
                param_score -= 1
            
            # Normalize score to [0, 1]
            param_score = max(0, min(1, (param_score + 2) / 4))
            
            quality_report['parameter_scores'][name] = {
                'score': param_score,
                'warnings': warnings
            }
            
            total_score += param_score
        
        quality_report['overall_score'] = total_score / num_params
        
        if quality_report['overall_score'] < 0.5:
            quality_report['recommendations'].append("Consider using different initialization method")
            quality_report['recommendations'].append("Check for gradient flow issues")
        
        return quality_report


class CustomInitializationSchemes:
    """Custom initialization schemes for specific architectures."""
    
    @staticmethod
    def transformer_initialization(model: nn.Module, d_model: int) -> None:
        """Initialize transformer model with appropriate schemes."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'attention' in name.lower() or 'attn' in name.lower():
                    WeightInitializer.transformer_init(module.weight, d_model)
                else:
                    WeightInitializer.transformer_init(module.weight, d_model)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.zeros_(module.bias)
            
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0, std=0.02)
            
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)
    
    @staticmethod
    def cnn_initialization(model: nn.Module) -> None:
        """Initialize CNN model with appropriate schemes."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                WeightInitializer.conv_init(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.zeros_(module.bias)
            
            elif isinstance(module, nn.Linear):
                WeightInitializer.xavier_uniform(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.zeros_(module.bias)
            
            elif isinstance(module, nn.BatchNorm2d):
                init.ones_(module.weight)
                init.zeros_(module.bias)
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
    
    @staticmethod
    def rnn_initialization(model: nn.Module, num_layers: int = 1) -> None:
        """Initialize RNN model with appropriate schemes."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        WeightInitializer.lstm_init(param, num_layers)
                    elif 'bias' in param_name:
                        init.zeros_(param)
            
            elif isinstance(module, nn.Linear):
                WeightInitializer.xavier_uniform(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.zeros_(module.bias)


def demonstrate_weight_initialization():
    """Demonstrate the weight initialization system."""
    print("Weight Initialization System Demonstration")
    print("=" * 50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    print("1. Analyzing uninitialized model...")
    analyzer = InitializationAnalyzer()
    stats = analyzer.analyze_weights(model)
    print(f"   Found {len(stats)} parameters")
    
    print("\n2. Applying Xavier uniform initialization...")
    WeightInitializer.initialize_model(model, 'xavier_uniform')
    
    print("\n3. Analyzing initialized model...")
    stats = analyzer.analyze_weights(model)
    for name, param_stats in stats.items():
        print(f"   {name}: μ={param_stats['mean']:.4f}, σ={param_stats['std']:.4f}")
    
    print("\n4. Checking initialization quality...")
    quality = analyzer.check_initialization_quality(model)
    print(f"   Overall score: {quality['overall_score']:.2f}")
    
    print("\nWeight initialization demonstration completed!")


if __name__ == "__main__":
    demonstrate_weight_initialization()

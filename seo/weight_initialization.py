from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import math
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Weight Initialization and Normalization Techniques for SEO Service
Advanced initialization strategies and normalization methods for optimal model performance
"""


logger = logging.getLogger(__name__)

@dataclass
class InitializationConfig:
    """Configuration for weight initialization strategies"""
    method: str = "xavier_uniform"  # xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal, sparse
    gain: float = 1.0
    fan_mode: str = "fan_in"  # fan_in, fan_out, fan_avg
    nonlinearity: str = "leaky_relu"  # relu, leaky_relu, tanh, sigmoid, linear
    sparsity: float = 0.1  # For sparse initialization
    std: float = 0.02  # For normal distribution initialization
    mean: float = 0.0  # For normal distribution initialization
    min_val: float = -0.1  # For uniform distribution bounds
    max_val: float = 0.1  # For uniform distribution bounds

@dataclass
class NormalizationConfig:
    """Configuration for normalization techniques"""
    method: str = "layer_norm"  # layer_norm, batch_norm, instance_norm, group_norm, weight_norm
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    num_groups: int = 32  # For group normalization
    elementwise_affine: bool = True  # For layer normalization

class AdvancedWeightInitializer:
    """Advanced weight initialization strategies"""
    
    @staticmethod
    def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0, fan_mode: str = "fan_avg") -> torch.Tensor:
        """Xavier uniform initialization with configurable fan mode"""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        
        if fan_mode == "fan_in":
            fan = fan_in
        elif fan_mode == "fan_out":
            fan = fan_out
        elif fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"Unsupported fan_mode: {fan_mode}")
        
        std = gain * math.sqrt(2.0 / fan)
        bound = math.sqrt(3.0) * std
        
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        
        return tensor
    
    @staticmethod
    def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, fan_mode: str = "fan_avg") -> torch.Tensor:
        """Xavier normal initialization with configurable fan mode"""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        
        if fan_mode == "fan_in":
            fan = fan_in
        elif fan_mode == "fan_out":
            fan = fan_out
        elif fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"Unsupported fan_mode: {fan_mode}")
        
        std = gain * math.sqrt(2.0 / fan)
        
        with torch.no_grad():
            tensor.normal_(0, std)
        
        return tensor
    
    @staticmethod
    def kaiming_uniform_(tensor: torch.Tensor, a: float = 0, mode: str = "fan_in", 
                        nonlinearity: str = "leaky_relu") -> torch.Tensor:
        """Kaiming uniform initialization with configurable nonlinearity"""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == "fan_in" else fan_out
        
        if nonlinearity == "linear" or nonlinearity == "conv1d" or nonlinearity == "conv2d" or nonlinearity == "conv3d" or nonlinearity == "conv_transpose1d" or nonlinearity == "conv_transpose2d" or nonlinearity == "conv_transpose3d":
            gain = 1
        elif nonlinearity == "sigmoid":
            gain = 1
        elif nonlinearity == "tanh":
            gain = 5.0 / 3
        elif nonlinearity == "relu":
            gain = math.sqrt(2.0)
        elif nonlinearity == "leaky_relu":
            gain = math.sqrt(2.0 / (1 + a ** 2))
        else:
            gain = 1
        
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        
        return tensor
    
    @staticmethod
    def kaiming_normal_(tensor: torch.Tensor, a: float = 0, mode: str = "fan_in", 
                       nonlinearity: str = "leaky_relu") -> torch.Tensor:
        """Kaiming normal initialization with configurable nonlinearity"""
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == "fan_in" else fan_out
        
        if nonlinearity == "linear" or nonlinearity == "conv1d" or nonlinearity == "conv2d" or nonlinearity == "conv3d" or nonlinearity == "conv_transpose1d" or nonlinearity == "conv_transpose2d" or nonlinearity == "conv_transpose3d":
            gain = 1
        elif nonlinearity == "sigmoid":
            gain = 1
        elif nonlinearity == "tanh":
            gain = 5.0 / 3
        elif nonlinearity == "relu":
            gain = math.sqrt(2.0)
        elif nonlinearity == "leaky_relu":
            gain = math.sqrt(2.0 / (1 + a ** 2))
        else:
            gain = 1
        
        std = gain / math.sqrt(fan)
        
        with torch.no_grad():
            tensor.normal_(0, std)
        
        return tensor
    
    @staticmethod
    def orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Orthogonal initialization for better gradient flow"""
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")
        
        rows = tensor.size(0)
        cols = tensor.numel() // rows
        flattened = tensor.new(rows, cols).normal_(0, 1)
        
        if rows < cols:
            flattened.t_()
        
        # Compute the qr factorization
        q, r = torch.linalg.qr(flattened)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph.unsqueeze(0)
        
        if rows < cols:
            q.t_()
        
        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)
        
        return tensor
    
    @staticmethod
    def sparse_(tensor: torch.Tensor, sparsity: float = 0.1, std: float = 0.01) -> torch.Tensor:
        """Sparse initialization for regularization"""
        with torch.no_grad():
            tensor.normal_(0, std)
            mask = torch.rand(tensor.size()) > sparsity
            tensor.mul_(mask)
        
        return tensor
    
    @staticmethod
    def delta_orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Delta orthogonal initialization for RNNs"""
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")
        
        rows = tensor.size(0)
        cols = tensor.numel() // rows
        
        flattened = tensor.new(rows, cols).normal_(0, 1)
        u, _, v = torch.svd(flattened, full_matrices=False)
        
        # Pick the one with the correct shape
        q = u if u.shape == flattened.shape else v
        q = q.view(tensor.shape)
        
        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)
        
        return tensor
    
    @staticmethod
    def init_weights(module: nn.Module, config: InitializationConfig) -> None:
        """Initialize weights of a module using specified configuration"""
        for name, param in module.named_parameters():
            if 'weight' in name:
                if config.method == "xavier_uniform":
                    AdvancedWeightInitializer.xavier_uniform_(param, config.gain, config.fan_mode)
                elif config.method == "xavier_normal":
                    AdvancedWeightInitializer.xavier_normal_(param, config.gain, config.fan_mode)
                elif config.method == "kaiming_uniform":
                    AdvancedWeightInitializer.kaiming_uniform_(param, mode=config.fan_mode, nonlinearity=config.nonlinearity)
                elif config.method == "kaiming_normal":
                    AdvancedWeightInitializer.kaiming_normal_(param, mode=config.fan_mode, nonlinearity=config.nonlinearity)
                elif config.method == "orthogonal":
                    AdvancedWeightInitializer.orthogonal_(param, config.gain)
                elif config.method == "sparse":
                    AdvancedWeightInitializer.sparse_(param, config.sparsity, config.std)
                elif config.method == "normal":
                    init.normal_(param, config.mean, config.std)
                elif config.method == "uniform":
                    init.uniform_(param, config.min_val, config.max_val)
                elif config.method == "constant":
                    init.constant_(param, config.mean)
                else:
                    raise ValueError(f"Unsupported initialization method: {config.method}")
                
                logger.info(f"Initialized {name} with {config.method}")
            
            elif 'bias' in name:
                if config.method in ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"]:
                    init.zeros_(param)
                else:
                    init.constant_(param, config.mean)
                
                logger.info(f"Initialized {name} bias")

class AdvancedNormalization:
    """Advanced normalization techniques"""
    
    @staticmethod
    def create_normalization_layer(config: NormalizationConfig, num_features: int) -> nn.Module:
        """Create normalization layer based on configuration"""
        if config.method == "layer_norm":
            return nn.LayerNorm(num_features, eps=config.eps, elementwise_affine=config.elementwise_affine)
        elif config.method == "batch_norm":
            return nn.BatchNorm1d(num_features, eps=config.eps, momentum=config.momentum, 
                                affine=config.affine, track_running_stats=config.track_running_stats)
        elif config.method == "instance_norm":
            return nn.InstanceNorm1d(num_features, eps=config.eps, momentum=config.momentum,
                                   affine=config.affine, track_running_stats=config.track_running_stats)
        elif config.method == "group_norm":
            return nn.GroupNorm(config.num_groups, num_features, eps=config.eps, affine=config.affine)
        else:
            raise ValueError(f"Unsupported normalization method: {config.method}")

class WeightNormLinear(nn.Module):
    """Linear layer with weight normalization"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 eps: float = 1e-5, init_scale: float = 1.0):
        
    """__init__ function."""
super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.init_scale = init_scale
        
        # Initialize weight and bias
        self.weight_v = Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = Parameter(torch.Tensor(out_features, 1))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> Any:
        """Initialize parameters"""
        init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        init.constant_(self.weight_g, self.init_scale)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_v)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight normalization"""
        # Normalize weight
        weight_norm = self.weight_v.norm(p=2, dim=1, keepdim=True)
        weight = self.weight_g * self.weight_v / (weight_norm + self.eps)
        
        return F.linear(input_tensor, weight, self.bias)

class AdaptiveWeightNorm(nn.Module):
    """Adaptive weight normalization with learnable scale"""
    
    def __init__(self, module: nn.Module, eps: float = 1e-5, init_scale: float = 1.0):
        
    """__init__ function."""
super().__init__()
        self.module = module
        self.eps = eps
        self.init_scale = init_scale
        
        # Add weight norm parameters
        for name, param in self.module.named_parameters():
            if 'weight' in name:
                # Store original weight
                setattr(self, f"{name}_v", param.data.clone())
                # Create scale parameter
                scale = torch.ones(param.size(0), 1) * self.init_scale
                setattr(self, f"{name}_g", Parameter(scale))
                # Replace weight with normalized version
                param.data = self._normalize_weight(param.data, scale)
    
    def _normalize_weight(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Normalize weight using scale parameter"""
        weight_norm = weight.norm(p=2, dim=1, keepdim=True)
        return scale * weight / (weight_norm + self.eps)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with adaptive weight normalization"""
        # Update weights with normalization
        for name, param in self.module.named_parameters():
            if 'weight' in name:
                weight_v = getattr(self, f"{name}_v")
                weight_g = getattr(self, f"{name}_g")
                param.data = self._normalize_weight(weight_v, weight_g)
        
        return self.module(*args, **kwargs)

class SpectralNorm(nn.Module):
    """Spectral normalization for improved training stability"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        
    """__init__ function."""
super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _update_u(self) -> Any:
        """Update u vector for spectral normalization"""
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = F.normalize(torch.randn(height).unsqueeze(0), dim=1, eps=1e-12)
            u = F.normalize(torch.mv(w.view(height, -1), v.t()), dim=0, eps=1e-12)
            v = F.normalize(torch.mv(w.view(height, -1).t(), u.unsqueeze(1)).squeeze(), dim=0, eps=1e-12)
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name + "_u", u)
        setattr(self.module, self.name + "_v", v)
        setattr(self.module, self.name + "_sigma", sigma)
    
    def _made_params(self) -> Any:
        """Check if spectral norm parameters exist"""
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w_bar = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self) -> Any:
        """Create spectral norm parameters"""
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = Parameter(w.data.new(height).normal_(0, 1))
        v = Parameter(w.data.new(width).normal_(0, 1))
        u.data = F.normalize(u.data, dim=0, eps=1e-12)
        v.data = F.normalize(v.data, dim=0, eps=1e-12)
        w_bar = Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with spectral normalization"""
        self._update_u()
        return self.module.forward(*args, **kwargs)

class WeightInitializationManager:
    """Manager for weight initialization strategies"""
    
    def __init__(self) -> Any:
        self.initialization_history = []
        self.normalization_history = []
    
    def initialize_model(self, model: nn.Module, config: InitializationConfig) -> None:
        """Initialize entire model with specified configuration"""
        logger.info(f"Initializing model with {config.method} method")
        
        # Initialize weights
        AdvancedWeightInitializer.init_weights(model, config)
        
        # Record initialization
        self.initialization_history.append({
            'method': config.method,
            'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None,
            'config': config
        })
        
        logger.info("Model initialization completed")
    
    def apply_normalization(self, model: nn.Module, config: NormalizationConfig) -> nn.Module:
        """Apply normalization to model layers"""
        logger.info(f"Applying {config.method} normalization")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with weight normalized version
                if config.method == "weight_norm":
                    new_module = WeightNormLinear(
                        module.in_features, 
                        module.out_features, 
                        bias=module.bias is not None
                    )
                    # Copy weights
                    new_module.weight_v.data = module.weight.data
                    if module.bias is not None:
                        new_module.bias.data = module.bias.data
                    
                    # Replace module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_module)
        
        self.normalization_history.append({
            'method': config.method,
            'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None,
            'config': config
        })
        
        return model
    
    def get_initialization_summary(self) -> Dict[str, Any]:
        """Get summary of initialization history"""
        return {
            'total_initializations': len(self.initialization_history),
            'methods_used': [init['method'] for init in self.initialization_history],
            'normalizations_applied': [norm['method'] for norm in self.normalization_history]
        }

class WeightAnalysis:
    """Analyze weight distributions and properties"""
    
    @staticmethod
    def analyze_weights(model: nn.Module) -> Dict[str, Any]:
        """Analyze weight distributions in model"""
        analysis = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_data = param.data.cpu().numpy()
                
                analysis[name] = {
                    'mean': float(np.mean(weight_data)),
                    'std': float(np.std(weight_data)),
                    'min': float(np.min(weight_data)),
                    'max': float(np.max(weight_data)),
                    'sparsity': float(np.sum(weight_data == 0) / weight_data.size),
                    'shape': list(weight_data.shape),
                    'norm_l2': float(np.linalg.norm(weight_data, ord=2)),
                    'norm_l1': float(np.linalg.norm(weight_data, ord=1))
                }
        
        return analysis
    
    @staticmethod
    def check_weight_health(model: nn.Module) -> Dict[str, Any]:
        """Check for potential weight-related issues"""
        issues = []
        warnings = []
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_data = param.data
                
                # Check for NaN or Inf
                if torch.isnan(weight_data).any():
                    issues.append(f"NaN values in {name}")
                
                if torch.isinf(weight_data).any():
                    issues.append(f"Infinite values in {name}")
                
                # Check for very large weights
                weight_norm = torch.norm(weight_data)
                if weight_norm > 100:
                    warnings.append(f"Large weight norm in {name}: {weight_norm:.4f}")
                
                # Check for dead neurons (all zero weights)
                if torch.all(weight_data == 0):
                    issues.append(f"All zero weights in {name}")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'is_healthy': len(issues) == 0
        }

# Utility functions
def initialize_weights_xavier(model: nn.Module, gain: float = 1.0) -> None:
    """Initialize model weights using Xavier initialization"""
    config = InitializationConfig(method="xavier_uniform", gain=gain)
    AdvancedWeightInitializer.init_weights(model, config)

def initialize_weights_kaiming(model: nn.Module, nonlinearity: str = "relu") -> None:
    """Initialize model weights using Kaiming initialization"""
    config = InitializationConfig(method="kaiming_uniform", nonlinearity=nonlinearity)
    AdvancedWeightInitializer.init_weights(model, config)

def initialize_weights_orthogonal(model: nn.Module, gain: float = 1.0) -> None:
    """Initialize model weights using orthogonal initialization"""
    config = InitializationConfig(method="orthogonal", gain=gain)
    AdvancedWeightInitializer.init_weights(model, config)

def apply_weight_norm(model: nn.Module) -> nn.Module:
    """Apply weight normalization to model"""
    config = NormalizationConfig(method="weight_norm")
    manager = WeightInitializationManager()
    return manager.apply_normalization(model, config)

def apply_spectral_norm(model: nn.Module) -> nn.Module:
    """Apply spectral normalization to model"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            setattr(model, name, SpectralNorm(module))
    return model

# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Initialize with different methods
    print("=== Weight Initialization Examples ===")
    
    # Xavier initialization
    initialize_weights_xavier(model)
    analysis = WeightAnalysis.analyze_weights(model)
    print("Xavier initialization analysis:", analysis)
    
    # Kaiming initialization
    model2 = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    initialize_weights_kaiming(model2)
    analysis2 = WeightAnalysis.analyze_weights(model2)
    print("Kaiming initialization analysis:", analysis2)
    
    # Weight health check
    health = WeightAnalysis.check_weight_health(model)
    print("Weight health check:", health) 
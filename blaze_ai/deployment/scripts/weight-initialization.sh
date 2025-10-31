#!/usr/bin/env python3
"""
Advanced Weight Initialization and Normalization for Blaze AI
Implements proper weight initialization schemes, normalization techniques, and best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import warnings
import math

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InitializationConfig:
    """Configuration for weight initialization"""
    method: str = "xavier_uniform"  # xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal, sparse
    gain: float = 1.0
    fan_mode: str = "fan_in"  # fan_in, fan_out, fan_avg
    nonlinearity: str = "leaky_relu"  # relu, leaky_relu, tanh, linear
    sparsity: float = 0.1
    std: float = 0.02
    mean: float = 0.0


@dataclass
class NormalizationConfig:
    """Configuration for normalization layers"""
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    num_groups: int = 32
    num_features: int = 512


class AdvancedWeightInitializer:
    """Advanced weight initialization with multiple schemes"""
    
    def __init__(self, config: InitializationConfig):
        self.config = config
        self.initialization_methods = {
            "xavier_uniform": self._xavier_uniform,
            "xavier_normal": self._xavier_normal,
            "kaiming_uniform": self._kaiming_uniform,
            "kaiming_normal": self._kaiming_normal,
            "orthogonal": self._orthogonal,
            "sparse": self._sparse,
            "truncated_normal": self._truncated_normal,
            "variance_scaling": self._variance_scaling,
            "lecun_normal": self._lecun_normal,
            "lecun_uniform": self._lecun_uniform
        }
    
    def initialize_weights(self, module: nn.Module) -> None:
        """Initialize weights for a module using the configured method"""
        
        method = self.initialization_methods.get(self.config.method)
        if method is None:
            raise ValueError(f"Unknown initialization method: {self.config.method}")
        
        for name, param in module.named_parameters():
            if 'weight' in name:
                method(param)
                logger.debug(f"Initialized {name} with {self.config.method}")
            elif 'bias' in name:
                init.constant_(param, 0.0)
                logger.debug(f"Initialized {name} with zeros")
    
    def _xavier_uniform(self, tensor: torch.Tensor) -> None:
        """Xavier/Glorot uniform initialization"""
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        std = self.config.gain * math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        init.uniform_(tensor, -bound, bound)
    
    def _xavier_normal(self, tensor: torch.Tensor) -> None:
        """Xavier/Glorot normal initialization"""
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        std = self.config.gain * math.sqrt(2.0 / (fan_in + fan_out))
        init.normal_(tensor, 0, std)
    
    def _kaiming_uniform(self, tensor: torch.Tensor) -> None:
        """Kaiming/He uniform initialization"""
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if self.config.fan_mode == "fan_in" else fan_out
        if self.config.fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        
        gain = self._calculate_gain(self.config.nonlinearity)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        init.uniform_(tensor, -bound, bound)
    
    def _kaiming_normal(self, tensor: torch.Tensor) -> None:
        """Kaiming/He normal initialization"""
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if self.config.fan_mode == "fan_in" else fan_out
        if self.config.fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        
        gain = self._calculate_gain(self.config.nonlinearity)
        std = gain / math.sqrt(fan)
        init.normal_(tensor, 0, std)
    
    def _orthogonal(self, tensor: torch.Tensor) -> None:
        """Orthogonal initialization"""
        init.orthogonal_(tensor, gain=self.config.gain)
    
    def _sparse(self, tensor: torch.Tensor) -> None:
        """Sparse initialization"""
        init.sparse_(tensor, sparsity=self.config.sparsity, std=self.config.std)
    
    def _truncated_normal(self, tensor: torch.Tensor) -> None:
        """Truncated normal initialization"""
        init.trunc_normal_(tensor, mean=self.config.mean, std=self.config.std, a=-2, b=2)
    
    def _variance_scaling(self, tensor: torch.Tensor) -> None:
        """Variance scaling initialization"""
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if self.config.fan_mode == "fan_in" else fan_out
        if self.config.fan_mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        
        scale = self.config.gain / math.sqrt(fan)
        init.uniform_(tensor, -scale, scale)
    
    def _lecun_normal(self, tensor: torch.Tensor) -> None:
        """LeCun normal initialization"""
        fan_in, _ = self._calculate_fan_in_and_fan_out(tensor)
        std = self.config.gain / math.sqrt(fan_in)
        init.normal_(tensor, 0, std)
    
    def _lecun_uniform(self, tensor: torch.Tensor) -> None:
        """LeCun uniform initialization"""
        fan_in, _ = self._calculate_fan_in_and_fan_out(tensor)
        bound = self.config.gain * math.sqrt(3.0 / fan_in)
        init.uniform_(tensor, -bound, bound)
    
    def _calculate_fan_in_and_fan_out(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Calculate fan_in and fan_out for a tensor"""
        if tensor.dim() < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        
        return fan_in, fan_out
    
    def _calculate_gain(self, nonlinearity: str) -> float:
        """Calculate gain for different nonlinearities"""
        linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 
                     'conv_transpose2d', 'conv_transpose3d']
        if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
            return 1
        elif nonlinearity == 'tanh':
            return 5.0 / 3
        elif nonlinearity == 'relu':
            return math.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            return math.sqrt(2.0 / (1 + 0.01 ** 2))
        else:
            return 1.0


class AdvancedNormalization:
    """Advanced normalization techniques"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
    
    def create_normalization_layer(self, layer_type: str, num_features: int) -> nn.Module:
        """Create normalization layer based on type"""
        
        if layer_type == "batch_norm":
            return nn.BatchNorm1d(num_features, eps=self.config.eps, 
                                 momentum=self.config.momentum, 
                                 affine=self.config.affine,
                                 track_running_stats=self.config.track_running_stats)
        
        elif layer_type == "layer_norm":
            return nn.LayerNorm(num_features, eps=self.config.eps, 
                              elementwise_affine=self.config.affine)
        
        elif layer_type == "instance_norm":
            return nn.InstanceNorm1d(num_features, eps=self.config.eps, 
                                   momentum=self.config.momentum,
                                   affine=self.config.affine,
                                   track_running_stats=self.config.track_running_stats)
        
        elif layer_type == "group_norm":
            return nn.GroupNorm(self.config.num_groups, num_features, 
                              eps=self.config.eps, affine=self.config.affine)
        
        elif layer_type == "rms_norm":
            return RMSNorm(num_features, eps=self.config.eps)
        
        elif layer_type == "adaptive_norm":
            return AdaptiveNorm(num_features, eps=self.config.eps)
        
        else:
            raise ValueError(f"Unknown normalization type: {layer_type}")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class AdaptiveNorm(nn.Module):
    """Adaptive Normalization with learnable parameters"""
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(1))  # Adaptive scaling
        self.beta = nn.Parameter(torch.zeros(1))  # Adaptive shifting
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute statistics
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply adaptive transformation
        x_adapted = self.alpha * x_norm + self.beta
        
        # Apply weight and bias
        return self.weight * x_adapted + self.bias


class WeightInitializationTester:
    """Test different weight initialization schemes"""
    
    def __init__(self):
        self.results = {}
    
    def test_initialization_methods(self, input_size: int = 100, hidden_size: int = 200) -> Dict[str, Any]:
        """Test various initialization methods"""
        
        methods = ["xavier_uniform", "xavier_normal", "kaiming_uniform", 
                  "kaiming_normal", "orthogonal", "truncated_normal"]
        
        for method in methods:
            logger.info(f"Testing {method} initialization...")
            
            # Create model
            model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            
            # Initialize weights
            config = InitializationConfig(method=method)
            initializer = AdvancedWeightInitializer(config)
            initializer.initialize_weights(model)
            
            # Analyze weight distributions
            weight_stats = self._analyze_weight_distribution(model)
            self.results[method] = weight_stats
            
            logger.info(f"{method} - Mean: {weight_stats['mean']:.4f}, "
                       f"Std: {weight_stats['std']:.4f}, "
                       f"Range: [{weight_stats['min']:.4f}, {weight_stats['max']:.4f}]")
        
        return self.results
    
    def _analyze_weight_distribution(self, model: nn.Module) -> Dict[str, float]:
        """Analyze weight distribution in a model"""
        
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.flatten())
        
        if not all_weights:
            return {}
        
        all_weights = torch.cat(all_weights)
        
        return {
            'mean': all_weights.mean().item(),
            'std': all_weights.std().item(),
            'min': all_weights.min().item(),
            'max': all_weights.max().item(),
            'var': all_weights.var().item(),
            'norm': all_weights.norm().item()
        }


class NormalizationTester:
    """Test different normalization techniques"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.normalizer = AdvancedNormalization(config)
        self.results = {}
    
    def test_normalization_methods(self, batch_size: int = 32, seq_len: int = 100, 
                                 hidden_size: int = 512) -> Dict[str, Any]:
        """Test various normalization methods"""
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        normalization_types = ["batch_norm", "layer_norm", "instance_norm", 
                             "group_norm", "rms_norm", "adaptive_norm"]
        
        for norm_type in normalization_types:
            logger.info(f"Testing {norm_type} normalization...")
            
            try:
                # Create normalization layer
                if norm_type == "batch_norm":
                    norm_layer = nn.BatchNorm1d(hidden_size, eps=self.config.eps,
                                              momentum=self.config.momentum,
                                              affine=self.config.affine)
                    # Reshape for BatchNorm1d
                    x_reshaped = x.transpose(1, 2).contiguous()  # (batch, hidden, seq)
                    output = norm_layer(x_reshaped)
                    output = output.transpose(1, 2).contiguous()  # (batch, seq, hidden)
                else:
                    norm_layer = self.normalizer.create_normalization_layer(norm_type, hidden_size)
                    output = norm_layer(x)
                
                # Analyze output
                output_stats = self._analyze_normalization_output(output)
                self.results[norm_type] = output_stats
                
                logger.info(f"{norm_type} - Mean: {output_stats['mean']:.4f}, "
                           f"Std: {output_stats['std']:.4f}, "
                           f"Range: [{output_stats['min']:.4f}, {output_stats['max']:.4f}]")
                
            except Exception as e:
                logger.error(f"Error testing {norm_type}: {e}")
                self.results[norm_type] = {'error': str(e)}
        
        return self.results
    
    def _analyze_normalization_output(self, output: torch.Tensor) -> Dict[str, float]:
        """Analyze normalization output statistics"""
        
        return {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item(),
            'var': output.var().item(),
            'norm': output.norm().item()
        }


class OptimizedModelArchitecture(nn.Module):
    """Model architecture with optimized weight initialization and normalization"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 init_config: InitializationConfig, norm_config: NormalizationConfig):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        input_layer = nn.Linear(input_size, hidden_size)
        self.layers.append(input_layer)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            hidden_layer = nn.Linear(hidden_size, hidden_size)
            self.layers.append(hidden_layer)
        
        # Output layer
        output_layer = nn.Linear(hidden_size, 1)
        self.layers.append(output_layer)
        
        # Normalization layers
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            norm_layer = nn.LayerNorm(hidden_size, eps=norm_config.eps)
            self.norm_layers.append(norm_layer)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._initialize_weights(init_config)
    
    def _initialize_weights(self, config: InitializationConfig) -> None:
        """Initialize weights using the specified method"""
        initializer = AdvancedWeightInitializer(config)
        initializer.initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization and activation"""
        
        # Input layer
        x = self.layers[0](x)
        x = self.norm_layers[0](x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Hidden layers
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        return x


class WeightInitializationBenchmark:
    """Benchmark different initialization methods"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def run_benchmark(self, input_size: int = 100, hidden_size: int = 200, 
                     num_layers: int = 5, num_iterations: int = 100) -> Dict[str, Any]:
        """Run benchmark comparing initialization methods"""
        
        methods = ["xavier_uniform", "xavier_normal", "kaiming_uniform", 
                  "kaiming_normal", "orthogonal"]
        
        for method in methods:
            logger.info(f"Benchmarking {method}...")
            
            # Create model
            model = OptimizedModelArchitecture(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                init_config=InitializationConfig(method=method),
                norm_config=NormalizationConfig()
            )
            
            # Generate random input
            x = torch.randn(32, input_size)
            
            # Measure forward pass time
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time and end_time:
                start_time.record()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = model(x)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
            else:
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time and end_time:
                    start_time.record()
                    for _ in range(num_iterations):
                        with torch.no_grad():
                            _ = model(x)
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)
                else:
                    import time
                    start_time = time.time()
                    for _ in range(num_iterations):
                        with torch.no_grad():
                            _ = model(x)
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Analyze weight distribution
            weight_stats = self._analyze_model_weights(model)
            
            self.benchmark_results[method] = {
                'elapsed_time': elapsed_time,
                'weight_stats': weight_stats,
                'num_parameters': sum(p.numel() for p in model.parameters())
            }
            
            logger.info(f"{method} - Time: {elapsed_time:.2f}ms, "
                       f"Params: {self.benchmark_results[method]['num_parameters']}")
        
        return self.benchmark_results
    
    def _analyze_model_weights(self, model: nn.Module) -> Dict[str, float]:
        """Analyze weight distribution in a model"""
        
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.flatten())
        
        if not all_weights:
            return {}
        
        all_weights = torch.cat(all_weights)
        
        return {
            'mean': all_weights.mean().item(),
            'std': all_weights.std().item(),
            'min': all_weights.min().item(),
            'max': all_weights.max().item(),
            'var': all_weights.var().item(),
            'norm': all_weights.norm().item()
        }


def main():
    """Main execution function"""
    logger.info("Starting Advanced Weight Initialization and Normalization Demonstrations...")
    
    # Test weight initialization methods
    logger.info("Testing weight initialization methods...")
    init_tester = WeightInitializationTester()
    init_results = init_tester.test_initialization_methods()
    
    # Test normalization methods
    logger.info("Testing normalization methods...")
    norm_config = NormalizationConfig()
    norm_tester = NormalizationTester(norm_config)
    norm_results = norm_tester.test_normalization_methods()
    
    # Run benchmark
    logger.info("Running initialization benchmark...")
    benchmark = WeightInitializationBenchmark()
    benchmark_results = benchmark.run_benchmark()
    
    # Create optimized model
    logger.info("Creating optimized model architecture...")
    init_config = InitializationConfig(method="kaiming_normal", nonlinearity="relu")
    norm_config = NormalizationConfig(eps=1e-5, momentum=0.1)
    
    model = OptimizedModelArchitecture(
        input_size=100,
        hidden_size=200,
        num_layers=5,
        init_config=init_config,
        norm_config=norm_config
    )
    
    # Test model
    x = torch.randn(32, 100)
    with torch.no_grad():
        output = model(x)
    
    logger.info(f"Model output shape: {output.shape}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Summary
    logger.info("Weight Initialization and Normalization Summary:")
    logger.info(f"Initialization methods tested: {len(init_results)}")
    logger.info(f"Normalization methods tested: {len(norm_results)}")
    logger.info(f"Benchmark methods: {len(benchmark_results)}")
    
    logger.info("Advanced Weight Initialization and Normalization demonstrations completed successfully!")


if __name__ == "__main__":
    main()

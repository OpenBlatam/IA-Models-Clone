from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
import time
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
PyTorch Autograd Examples
Comprehensive examples demonstrating automatic differentiation and gradient computation.
"""



@dataclass
class AutogradConfig:
    """Configuration for autograd examples."""
    input_size: int = 100
    hidden_size: int = 50
    output_size: int = 10
    batch_size: int = 32
    learning_rate: float = 0.01
    num_iterations: int = 1000
    use_gpu: bool = True
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0


class CustomAutogradFunction(autograd.Function):
    """Custom autograd function demonstrating manual gradient computation."""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient computation."""
        # Save tensors for backward pass
        ctx.save_for_backward(input_tensor, weight, bias)
        
        # Custom forward computation
        output = torch.matmul(input_tensor, weight.t()) + bias
        
        # Apply custom activation
        output = torch.where(
            output > 0,
            output * torch.exp(-output),
            torch.zeros_like(output)
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass with manual gradient computation."""
        # Retrieve saved tensors
        input_tensor, weight, bias = ctx.saved_tensors
        
        # Compute gradients manually
        # For the custom activation: f(x) = x * exp(-x) if x > 0, else 0
        # The derivative is: f'(x) = (1 - x) * exp(-x) if x > 0, else 0
        
        # Forward computation again for gradient
        linear_output = torch.matmul(input_tensor, weight.t()) + bias
        
        # Compute activation gradient
        activation_grad = torch.where(
            linear_output > 0,
            (1 - linear_output) * torch.exp(-linear_output),
            torch.zeros_like(linear_output)
        )
        
        # Apply gradient
        grad_linear = grad_output * activation_grad
        
        # Compute gradients for each input
        grad_input = torch.matmul(grad_linear, weight)
        grad_weight = torch.matmul(grad_linear.t(), input_tensor)
        grad_bias = grad_linear.sum(dim=0)
        
        return grad_input, grad_weight, grad_bias


class AutogradLinearLayer(nn.Module):
    """Linear layer with custom autograd function."""
    
    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        
    """__init__ function."""
super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom autograd function."""
        if self.bias is not None:
            return CustomAutogradFunction.apply(input_tensor, self.weight, self.bias)
        else:
            return CustomAutogradFunction.apply(input_tensor, self.weight, torch.zeros(self.output_features))


class GradientMonitoringModule(nn.Module):
    """Module for monitoring gradients during training."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        
    """__init__ function."""
super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Layers
        self.layer1 = AutogradLinearLayer(input_size, hidden_size)
        self.layer2 = AutogradLinearLayer(hidden_size, hidden_size)
        self.layer3 = AutogradLinearLayer(hidden_size, output_size)
        
        # Gradient monitoring
        self.gradient_norms = []
        self.gradient_means = []
        self.gradient_stds = []
        
        # Register hooks
        self._register_gradient_hooks()
    
    def _register_gradient_hooks(self) -> Any:
        """Register gradient hooks for monitoring."""
        for name, module in self.named_modules():
            if isinstance(module, AutogradLinearLayer):
                module.register_backward_hook(self._gradient_hook)
    
    def _gradient_hook(self, module, grad_input, grad_output) -> Any:
        """Gradient hook for monitoring."""
        if grad_input[0] is not None:
            grad_norm = grad_input[0].norm().item()
            grad_mean = grad_input[0].mean().item()
            grad_std = grad_input[0].std().item()
            
            self.gradient_norms.append(grad_norm)
            self.gradient_means.append(grad_mean)
            self.gradient_stds.append(grad_std)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient monitoring."""
        hidden1 = F.relu(self.layer1(input_tensor))
        hidden2 = F.relu(self.layer2(hidden1))
        output = self.layer3(hidden2)
        
        return output
    
    def get_gradient_stats(self) -> Dict[str, List[float]]:
        """Get gradient statistics."""
        return {
            'norms': self.gradient_norms,
            'means': self.gradient_means,
            'stds': self.gradient_stds
        }


class AutogradOptimizationExample:
    """Example demonstrating autograd optimization."""
    
    def __init__(self, config: AutogradConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = GradientMonitoringModule(
            config.input_size, config.hidden_size, config.output_size
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.losses = []
        self.gradient_norms = []
    
    def train_step(self, input_data: torch.Tensor, target_data: torch.Tensor) -> float:
        """Single training step with autograd."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(input_data)
        loss = self.criterion(output, target_data)
        
        # Backward pass (autograd computes gradients automatically)
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Compute gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item(), total_norm
    
    def train(self, num_iterations: int = None) -> Dict[str, List[float]]:
        """Training loop with autograd monitoring."""
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        for iteration in range(num_iterations):
            # Generate random data
            input_data = torch.randn(self.config.batch_size, self.config.input_size).to(self.device)
            target_data = torch.randn(self.config.batch_size, self.config.output_size).to(self.device)
            
            # Training step
            loss, grad_norm = self.train_step(input_data, target_data)
            
            # Record metrics
            self.losses.append(loss)
            self.gradient_norms.append(grad_norm)
            
            # Log progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}")
        
        return {
            'losses': self.losses,
            'gradient_norms': self.gradient_norms,
            'gradient_stats': self.model.get_gradient_stats()
        }


class AutogradCheckpointingExample:
    """Example demonstrating gradient checkpointing for memory efficiency."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 10):
        
    """__init__ function."""
self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create deep network
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, 
                     hidden_size if i < num_layers - 1 else input_size)
            for i in range(num_layers)
        ])
        
        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward_with_checkpointing(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        hidden = input_tensor
        
        for i, layer in enumerate(self.layers[:-1]):
            # Use checkpointing for intermediate layers
            if i < len(self.layers) - 2:  # Don't checkpoint last layer
                hidden = checkpoint(self._layer_forward, layer, hidden)
            else:
                hidden = self._layer_forward(layer, hidden)
        
        # Final layer
        output = self.layers[-1](hidden)
        return output
    
    def _layer_forward(self, layer: nn.Linear, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single layer."""
        return F.relu(layer(input_tensor))
    
    def forward_without_checkpointing(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass without gradient checkpointing."""
        hidden = input_tensor
        
        for layer in self.layers[:-1]:
            hidden = self._layer_forward(layer, hidden)
        
        output = self.layers[-1](hidden)
        return output


class AutogradGradientFlowExample:
    """Example demonstrating gradient flow analysis."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        
    """__init__ function."""
self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create network with different initialization strategies
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Initialize with different strategies
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights with different strategies."""
        # Xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        
        # Zero bias
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient flow monitoring."""
        hidden1 = F.relu(self.layer1(input_tensor))
        hidden2 = F.relu(self.layer2(hidden1))
        output = self.layer3(hidden2)
        
        return output
    
    def analyze_gradient_flow(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Analyze gradient flow through the network."""
        # Forward pass
        output = self.forward(input_tensor)
        
        # Create dummy loss
        target = torch.randn_like(output)
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        gradient_analysis = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                gradient_analysis[f"{name}_norm"] = grad_norm
                gradient_analysis[f"{name}_mean"] = grad_mean
                gradient_analysis[f"{name}_std"] = grad_std
        
        return gradient_analysis


def demonstrate_autograd_features():
    """Demonstrate various autograd features."""
    print("PyTorch Autograd Demonstration")
    print("=" * 50)
    
    # Configuration
    config = AutogradConfig(
        input_size=50,
        hidden_size=25,
        output_size=10,
        batch_size=16,
        learning_rate=0.01,
        num_iterations=500
    )
    
    # 1. Basic autograd optimization
    print("\n1. Basic Autograd Optimization")
    optimizer_example = AutogradOptimizationExample(config)
    results = optimizer_example.train()
    
    print(f"Final loss: {results['losses'][-1]:.6f}")
    print(f"Final gradient norm: {results['gradient_norms'][-1]:.6f}")
    
    # 2. Gradient checkpointing
    print("\n2. Gradient Checkpointing")
    checkpointing_example = AutogradCheckpointingExample(20, 30, 8)
    
    input_data = torch.randn(4, 20, requires_grad=True)
    
    # Without checkpointing
    start_time = time.time()
    output_without = checkpointing_example.forward_without_checkpointing(input_data)
    loss_without = output_without.sum()
    loss_without.backward()
    time_without = time.time() - start_time
    
    # With checkpointing
    start_time = time.time()
    output_with = checkpointing_example.forward_with_checkpointing(input_data)
    loss_with = output_with.sum()
    loss_with.backward()
    time_with = time.time() - start_time
    
    print(f"Time without checkpointing: {time_without:.4f}s")
    print(f"Time with checkpointing: {time_with:.4f}s")
    
    # 3. Gradient flow analysis
    print("\n3. Gradient Flow Analysis")
    flow_example = AutogradGradientFlowExample(30, 20, 10)
    input_data = torch.randn(8, 30, requires_grad=True)
    
    gradient_analysis = flow_example.analyze_gradient_flow(input_data)
    
    print("Gradient Analysis:")
    for key, value in gradient_analysis.items():
        print(f"  {key}: {value:.6f}")
    
    # 4. Custom autograd function
    print("\n4. Custom Autograd Function")
    custom_layer = AutogradLinearLayer(10, 5)
    input_data = torch.randn(4, 10, requires_grad=True)
    
    output = custom_layer(input_data)
    loss = output.sum()
    loss.backward()
    
    print(f"Custom autograd output shape: {output.shape}")
    print(f"Custom autograd loss: {loss.item():.6f}")
    
    return results, gradient_analysis


def demonstrate_gradient_computation():
    """Demonstrate manual gradient computation vs autograd."""
    print("\nGradient Computation Comparison")
    print("=" * 40)
    
    # Create simple function: f(x) = x^2 + 2x + 1
    def manual_function(x) -> Any:
        return x**2 + 2*x + 1
    
    def manual_gradient(x) -> Any:
        return 2*x + 2
    
    # Test with autograd
    x_autograd = torch.tensor([2.0], requires_grad=True)
    y_autograd = manual_function(x_autograd)
    y_autograd.backward()
    
    # Test with manual computation
    x_manual = torch.tensor([2.0])
    y_manual = manual_function(x_manual)
    grad_manual = manual_gradient(x_manual)
    
    print(f"Input: {x_autograd.item()}")
    print(f"Function value (autograd): {y_autograd.item()}")
    print(f"Function value (manual): {y_manual.item()}")
    print(f"Gradient (autograd): {x_autograd.grad.item()}")
    print(f"Gradient (manual): {grad_manual.item()}")
    
    # Verify they match
    assert abs(x_autograd.grad.item() - grad_manual.item()) < 1e-6
    print("✓ Autograd and manual gradients match!")


if __name__ == "__main__":
    # Run demonstrations
    results, gradient_analysis = demonstrate_autograd_features()
    demonstrate_gradient_computation()
    
    print("\nAutograd demonstration completed successfully!") 
"""
PyTorch Autograd System for Automatic Differentiation

This module provides comprehensive utilities and examples for utilizing PyTorch's autograd
system for automatic differentiation in deep learning models. It includes:

1. Autograd utilities for gradient computation
2. Custom loss functions with automatic differentiation
3. Gradient monitoring and debugging tools
4. Advanced autograd features (higher-order gradients, custom gradients)
5. Integration with the custom model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable, Function
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings


class AutogradUtils:
    """Utility class for PyTorch autograd operations and debugging."""
    
    @staticmethod
    def enable_gradients(model: nn.Module, enable: bool = True) -> None:
        """Enable or disable gradients for all parameters in a model."""
        for param in model.parameters():
            param.requires_grad = enable
    
    @staticmethod
    def get_gradient_norms(model: nn.Module) -> Dict[str, float]:
        """Get L2 norm of gradients for each parameter group."""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.norm().item()
        return gradients
    
    @staticmethod
    def check_gradients_exist(model: nn.Module) -> bool:
        """Check if any parameters have gradients."""
        return any(param.grad is not None for param in model.parameters())
    
    @staticmethod
    def zero_gradients(model: nn.Module) -> None:
        """Zero all gradients in the model."""
        model.zero_grad()
    
    @staticmethod
    def compute_gradient_statistics(model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Compute comprehensive gradient statistics."""
        stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                stats[name] = {
                    'mean': grad_data.mean().item(),
                    'std': grad_data.std().item(),
                    'min': grad_data.min().item(),
                    'max': grad_data.max().item(),
                    'norm': grad_data.norm().item(),
                    'norm_l1': grad_data.norm(1).item()
                }
        return stats


class CustomLossFunction(nn.Module):
    """Custom loss function that demonstrates autograd capabilities."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function with learnable parameters.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Computed loss value
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # L1 regularization on predictions
        l1_loss = torch.mean(torch.abs(predictions))
        
        # Total loss with learnable weights
        total_loss = self.alpha * mse_loss + self.beta * l1_loss
        
        return total_loss


class GradientMonitor:
    """Monitor and analyze gradients during training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = []
        self.parameter_history = []
    
    def register_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """Register hooks to monitor gradients."""
        hooks = []
        
        def gradient_hook(name):
            def hook(grad):
                if grad is not None:
                    self.gradient_history.append({
                        'name': name,
                        'grad_norm': grad.norm().item(),
                        'grad_mean': grad.mean().item(),
                        'grad_std': grad.std().item(),
                        'step': len(self.gradient_history)
                    })
            return hook
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(gradient_hook(name))
                hooks.append(hook)
        
        return hooks
    
    def get_gradient_summary(self) -> Dict[str, float]:
        """Get summary statistics of gradients."""
        if not self.gradient_history:
            return {}
        
        grad_norms = [entry['grad_norm'] for entry in self.gradient_history]
        grad_means = [entry['grad_mean'] for entry in self.gradient_history]
        grad_stds = [entry['grad_std'] for entry in self.gradient_history]
        
        return {
            'mean_grad_norm': np.mean(grad_norms),
            'std_grad_norm': np.std(grad_norms),
            'max_grad_norm': np.max(grad_norms),
            'min_grad_norm': np.min(grad_norms),
            'mean_grad_mean': np.mean(grad_means),
            'mean_grad_std': np.mean(grad_stds)
        }


class CustomAutogradFunction(Function):
    """Custom autograd function demonstrating manual gradient computation."""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of custom function.
        
        Args:
            input_tensor: Input tensor
            weight: Weight parameter
            
        Returns:
            Transformed tensor
        """
        ctx.save_for_backward(input_tensor, weight)
        return torch.tanh(input_tensor @ weight.T)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass with custom gradient computation.
        
        Args:
            grad_output: Gradient of output
            
        Returns:
            Gradients with respect to inputs
        """
        input_tensor, weight = ctx.saved_tensors
        
        # Custom gradient computation
        grad_input = grad_output * (1 - torch.tanh(input_tensor @ weight.T) ** 2) @ weight
        grad_weight = grad_output * (1 - torch.tanh(input_tensor @ weight.T) ** 2).unsqueeze(-1) * input_tensor.unsqueeze(-1)
        
        return grad_input, grad_weight.sum(dim=0)


class AutogradTrainingSystem:
    """Training system that leverages PyTorch autograd capabilities."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 loss_fn: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.gradient_monitor = GradientMonitor(model)
        self.hooks = []
        
        # Enable autograd for all parameters
        AutogradUtils.enable_gradients(self.model, True)
    
    def setup_gradient_monitoring(self):
        """Setup gradient monitoring hooks."""
        self.hooks = self.gradient_monitor.register_hooks()
    
    def cleanup_gradient_monitoring(self):
        """Remove gradient monitoring hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def train_step(self, batch_data: torch.Tensor, batch_targets: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with autograd.
        
        Args:
            batch_data: Input data batch
            batch_targets: Target labels batch
            
        Returns:
            Dictionary containing loss and gradient statistics
        """
        # Move data to device
        batch_data = batch_data.to(self.device)
        batch_targets = batch_targets.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(batch_data)
        
        # Compute loss
        loss = self.loss_fn(predictions, batch_targets)
        
        # Backward pass (autograd computes gradients)
        loss.backward()
        
        # Get gradient statistics
        grad_stats = AutogradUtils.compute_gradient_statistics(self.model)
        
        # Update parameters
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'gradient_stats': grad_stats,
            'gradient_summary': self.gradient_monitor.get_gradient_summary()
        }
    
    def compute_higher_order_gradients(self, input_data: torch.Tensor, 
                                     target_data: torch.Tensor, 
                                     order: int = 2) -> List[torch.Tensor]:
        """
        Compute higher-order gradients using autograd.
        
        Args:
            input_data: Input data
            target_data: Target data
            order: Order of gradients to compute
            
        Returns:
            List of gradients of different orders
        """
        input_data = input_data.to(self.device)
        target_data = target_data.to(self.device)
        
        # First forward pass
        predictions = self.model(input_data)
        loss = self.loss_fn(predictions, target_data)
        
        gradients = []
        current_grad = loss
        
        for i in range(order):
            # Compute gradients
            grads = torch.autograd.grad(
                current_grad, 
                self.model.parameters(), 
                create_graph=True,
                retain_graph=True
            )
            
            # Compute gradient norm
            grad_norm = sum(grad.norm() ** 2 for grad in grads).sqrt()
            gradients.append(grad_norm)
            
            # Prepare for next iteration
            current_grad = grad_norm
        
        return gradients


class AutogradExamples:
    """Examples demonstrating various autograd capabilities."""
    
    @staticmethod
    def basic_autograd_example():
        """Basic example of PyTorch autograd."""
        print("=== Basic Autograd Example ===")
        
        # Create tensors with requires_grad=True
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch.tensor([1.0, 2.0], requires_grad=True)
        
        # Define computation
        z = x ** 2 + y ** 3
        result = z.sum()
        
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"z: {z}")
        print(f"result: {result}")
        
        # Compute gradients
        result.backward()
        
        print(f"∂result/∂x: {x.grad}")
        print(f"∂result/∂y: {y.grad}")
        print()
    
    @staticmethod
    def custom_function_example():
        """Example using custom autograd function."""
        print("=== Custom Autograd Function Example ===")
        
        # Create custom function
        custom_fn = CustomAutogradFunction.apply
        
        # Create input tensors
        input_tensor = torch.randn(3, 2, requires_grad=True)
        weight = torch.randn(2, 4, requires_grad=True)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Weight shape: {weight.shape}")
        
        # Forward pass
        output = custom_fn(input_tensor, weight)
        print(f"Output shape: {output.shape}")
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        print(f"Input gradients shape: {input_tensor.grad.shape}")
        print(f"Weight gradients shape: {weight.grad.shape}")
        print()
    
    @staticmethod
    def gradient_flow_example():
        """Example demonstrating gradient flow through a simple network."""
        print("=== Gradient Flow Example ===")
        
        # Simple network
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
        # Create data
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)
        
        # Forward pass
        predictions = model(x)
        loss = F.mse_loss(predictions, y)
        
        print(f"Loss: {loss.item()}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad_norm = {param.grad.norm().item():.4f}")
        
        print()
    
    @staticmethod
    def higher_order_gradients_example():
        """Example of computing higher-order gradients."""
        print("=== Higher-Order Gradients Example ===")
        
        # Simple function
        x = torch.tensor(2.0, requires_grad=True)
        y = x ** 3 + 2 * x ** 2 + x
        
        # First derivative
        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
        print(f"dy/dx = {dy_dx.item()}")
        
        # Second derivative
        d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
        print(f"d²y/dx² = {d2y_dx2.item()}")
        
        # Third derivative
        d3y_dx3 = torch.autograd.grad(d2y_dx2, x)[0]
        print(f"d³y/dx³ = {d3y_dx3.item()}")
        print()


def demonstrate_autograd_system():
    """Demonstrate the complete autograd system."""
    print("PyTorch Autograd System Demonstration")
    print("=" * 50)
    
    # Run examples
    AutogradExamples.basic_autograd_example()
    AutogradExamples.custom_function_example()
    AutogradExamples.gradient_flow_example()
    AutogradExamples.higher_order_gradients_example()
    
    print("Autograd system demonstration completed!")


if __name__ == "__main__":
    demonstrate_autograd_system()



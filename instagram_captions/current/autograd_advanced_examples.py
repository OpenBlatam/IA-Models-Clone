"""
Advanced PyTorch Autograd Examples

This module provides advanced examples demonstrating PyTorch's autograd capabilities
integrated with the custom model architectures. It includes:

1. Advanced gradient computation techniques
2. Integration with custom model architectures
3. Sophisticated loss functions with autograd
4. Gradient analysis and visualization
5. Performance optimization with autograd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable, Function
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
import warnings

# Import custom model architectures
try:
    from custom_model_architectures import (
        CustomTransformerModel, CustomCNNModel, CustomRNNModel, 
        CNNTransformerHybrid, create_model_from_config
    )
except ImportError:
    print("Warning: custom_model_architectures not found. Some examples may not work.")
    CustomTransformerModel = CustomCNNModel = CustomRNNModel = CNNTransformerHybrid = None

# Import autograd system
try:
    from pytorch_autograd_system import (
        AutogradUtils, CustomLossFunction, GradientMonitor, 
        CustomAutogradFunction, AutogradTrainingSystem
    )
except ImportError:
    print("Warning: pytorch_autograd_system not found. Some examples may not work.")
    AutogradUtils = CustomLossFunction = GradientMonitor = CustomAutogradFunction = AutogradTrainingSystem = None


class AdvancedAutogradExamples:
    """Advanced examples demonstrating sophisticated autograd techniques."""
    
    @staticmethod
    def demonstrate_gradient_flow_analysis():
        """Demonstrate comprehensive gradient flow analysis through a network."""
        print("=== Gradient Flow Analysis ===")
        
        # Create a simple network
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.Softmax(dim=1)
        )
        
        # Create sample data
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        y_onehot = F.one_hot(y, num_classes=5).float()
        
        # Forward pass
        predictions = model(x)
        loss = F.cross_entropy(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradient flow
        print("Gradient flow analysis:")
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                relative_grad = grad_norm / (param_norm + 1e-8)
                
                print(f"  {name}:")
                print(f"    Gradient norm: {grad_norm:.6f}")
                print(f"    Parameter norm: {param_norm:.6f}")
                print(f"    Relative gradient: {relative_grad:.6f}")
        
        print()
    
    @staticmethod
    def demonstrate_custom_gradient_functions():
        """Demonstrate custom gradient functions with complex operations."""
        print("=== Custom Gradient Functions ===")
        
        class ComplexCustomFunction(Function):
            @staticmethod
            def forward(ctx, x, y, z):
                # Save tensors for backward pass
                ctx.save_for_backward(x, y, z)
                
                # Complex forward computation
                a = torch.sin(x) * torch.cos(y)
                b = torch.exp(-z)
                result = a * b + torch.tanh(x + y)
                
                return result
            
            @staticmethod
            def backward(ctx, grad_output):
                x, y, z = ctx.saved_tensors
                
                # Compute gradients manually
                grad_x = grad_output * (torch.cos(x) * torch.cos(y) + 
                                      (1 - torch.tanh(x + y) ** 2))
                grad_y = grad_output * (-torch.sin(x) * torch.sin(y) + 
                                      (1 - torch.tanh(x + y) ** 2))
                grad_z = grad_output * (-torch.exp(-z))
                
                return grad_x, grad_y, grad_z
        
        # Test custom function
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([0.5, 1.5], requires_grad=True)
        z = torch.tensor([0.1, 0.2], requires_grad=True)
        
        custom_fn = ComplexCustomFunction.apply
        result = custom_fn(x, y, z)
        
        print(f"Input x: {x}")
        print(f"Input y: {y}")
        print(f"Input z: {z}")
        print(f"Result: {result}")
        
        # Compute gradients
        loss = result.sum()
        loss.backward()
        
        print(f"Gradient ∂loss/∂x: {x.grad}")
        print(f"Gradient ∂loss/∂y: {y.grad}")
        print(f"Gradient ∂loss/∂z: {z.grad}")
        print()
    
    @staticmethod
    def demonstrate_higher_order_optimization():
        """Demonstrate higher-order optimization using autograd."""
        print("=== Higher-Order Optimization ===")
        
        # Define a complex function
        def complex_function(x, y):
            return torch.sin(x) * torch.exp(y) + torch.cos(x + y) ** 2
        
        # Create variables
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        
        # Compute function value
        f = complex_function(x, y)
        
        # First derivatives
        df_dx = grad(f, x, create_graph=True)[0]
        df_dy = grad(f, y, create_graph=True)[0]
        
        # Second derivatives
        d2f_dx2 = grad(df_dx, x, create_graph=True)[0]
        d2f_dy2 = grad(df_dy, y, create_graph=True)[0]
        d2f_dxdy = grad(df_dx, y, create_graph=True)[0]
        
        # Third derivatives
        d3f_dx3 = grad(d2f_dx2, x)[0]
        d3f_dy3 = grad(d2f_dy2, y)[0]
        
        print(f"Function value: {f.item():.6f}")
        print(f"First derivatives:")
        print(f"  ∂f/∂x = {df_dx.item():.6f}")
        print(f"  ∂f/∂y = {df_dy.item():.6f}")
        print(f"Second derivatives:")
        print(f"  ∂²f/∂x² = {d2f_dx2.item():.6f}")
        print(f"  ∂²f/∂y² = {d2f_dy2.item():.6f}")
        print(f"  ∂²f/∂x∂y = {d2f_dxdy.item():.6f}")
        print(f"Third derivatives:")
        print(f"  ∂³f/∂x³ = {d3f_dx3.item():.6f}")
        print(f"  ∂³f/∂y³ = {d3f_dy3.item():.6f}")
        print()
    
    @staticmethod
    def demonstrate_gradient_based_analysis():
        """Demonstrate gradient-based analysis techniques."""
        print("=== Gradient-Based Analysis ===")
        
        # Create a model
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        
        # Create data
        x = torch.randn(16, 5)
        y = torch.randn(16, 3)
        
        # Forward pass
        predictions = model(x)
        loss = F.mse_loss(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        print("Gradient analysis:")
        
        # Gradient statistics
        all_gradients = []
        for param in model.parameters():
            if param.grad is not None:
                all_gradients.append(param.grad.flatten())
        
        if all_gradients:
            all_gradients = torch.cat(all_gradients)
            
            print(f"  Total gradient elements: {len(all_gradients)}")
            print(f"  Mean gradient: {all_gradients.mean().item():.6f}")
            print(f"  Std gradient: {all_gradients.std().item():.6f}")
            print(f"  Min gradient: {all_gradients.min().item():.6f}")
            print(f"  Max gradient: {all_gradients.max().item():.6f}")
            
            # Check for gradient issues
            nan_count = torch.isnan(all_gradients).sum().item()
            inf_count = torch.isinf(all_gradients).sum().item()
            
            print(f"  NaN gradients: {nan_count}")
            print(f"  Inf gradients: {inf_count}")
            
            # Gradient distribution analysis
            grad_norm = all_gradients.norm().item()
            print(f"  Total gradient norm: {grad_norm:.6f}")
            
            if grad_norm > 10.0:
                print("  Warning: Large gradients detected!")
            elif grad_norm < 1e-6:
                print("  Warning: Very small gradients detected!")
        
        print()
    
    @staticmethod
    def demonstrate_autograd_with_custom_models():
        """Demonstrate autograd with custom model architectures."""
        if not all([CustomTransformerModel, CustomCNNModel, CustomRNNModel]):
            print("Warning: Custom models not available. Skipping this example.")
            return
        
        print("=== Autograd with Custom Models ===")
        
        # Create custom models
        transformer = CustomTransformerModel(
            vocab_size=1000,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512
        )
        
        cnn = CustomCNNModel(
            input_channels=3,
            num_classes=10,
            base_channels=32
        )
        
        rnn = CustomRNNModel(
            input_size=100,
            hidden_size=128,
            num_layers=2,
            num_classes=5
        )
        
        models = {
            'Transformer': transformer,
            'CNN': cnn,
            'RNN': rnn
        }
        
        # Test autograd for each model
        for name, model in models.items():
            print(f"\nTesting {name} model:")
            
            # Create sample data
            if name == 'Transformer':
                x = torch.randint(0, 1000, (8, 16))  # (batch, seq_len)
                y = torch.randint(0, 1000, (8, 16))
            elif name == 'CNN':
                x = torch.randn(8, 3, 32, 32)  # (batch, channels, height, width)
                y = torch.randint(0, 10, (8,))
            else:  # RNN
                x = torch.randn(8, 16, 100)  # (batch, seq_len, input_size)
                y = torch.randint(0, 5, (8,))
            
            # Forward pass
            try:
                predictions = model(x)
                
                # Compute loss
                if name == 'CNN' or name == 'RNN':
                    loss = F.cross_entropy(predictions, y)
                else:
                    loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), y.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                
                if grad_norms:
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
                    print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
                    print(f"  Parameters with gradients: {len(grad_norms)}")
                else:
                    print(f"  No gradients computed")
                
                # Zero gradients for next iteration
                model.zero_grad()
                
            except Exception as e:
                print(f"  Error testing {name}: {e}")
        
        print()
    
    @staticmethod
    def demonstrate_performance_optimization():
        """Demonstrate autograd performance optimization techniques."""
        print("=== Autograd Performance Optimization ===")
        
        # Create a large model for testing
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )
        
        # Create data
        x = torch.randn(64, 1000)
        y = torch.randn(64, 100)
        
        # Benchmark different approaches
        print("Benchmarking different approaches:")
        
        # Standard approach
        start_time = time.time()
        for _ in range(10):
            model.zero_grad()
            predictions = model(x)
            loss = F.mse_loss(predictions, y)
            loss.backward()
        
        standard_time = time.time() - start_time
        print(f"  Standard approach: {standard_time:.4f}s")
        
        # With gradient checkpointing
        model_checkpoint = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )
        
        # Enable gradient checkpointing
        for module in model_checkpoint.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
        
        start_time = time.time()
        for _ in range(10):
            model_checkpoint.zero_grad()
            predictions = model_checkpoint(x)
            loss = F.mse_loss(predictions, y)
            loss.backward()
        
        checkpoint_time = time.time() - start_time
        print(f"  With gradient checkpointing: {checkpoint_time:.4f}s")
        
        # Memory usage comparison
        print(f"  Memory efficiency: {standard_time/checkpoint_time:.2f}x")
        print()
    
    @staticmethod
    def demonstrate_gradient_visualization():
        """Demonstrate gradient visualization techniques."""
        print("=== Gradient Visualization ===")
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
        # Create data
        x = torch.randn(20, 2)
        y = torch.randn(20, 1)
        
        # Track gradients over multiple steps
        gradient_history = []
        
        for step in range(5):
            model.zero_grad()
            predictions = model(x)
            loss = F.mse_loss(predictions, y)
            loss.backward()
            
            # Collect gradient norms
            step_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    step_gradients.append(param.grad.norm().item())
            
            if step_gradients:
                gradient_history.append({
                    'step': step,
                    'mean_grad': np.mean(step_gradients),
                    'max_grad': np.max(step_gradients),
                    'min_grad': np.min(step_gradients)
                })
        
        # Print gradient history
        print("Gradient history:")
        for entry in gradient_history:
            print(f"  Step {entry['step']}: "
                  f"mean={entry['mean_grad']:.6f}, "
                  f"max={entry['max_grad']:.6f}, "
                  f"min={entry['min_grad']:.6f}")
        
        print()
    
    @staticmethod
    def demonstrate_autograd_debugging():
        """Demonstrate autograd debugging techniques."""
        print("=== Autograd Debugging ===")
        
        # Create a model with potential issues
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Create data
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        try:
            # Forward pass
            predictions = model(x)
            loss = F.mse_loss(predictions, y)
            
            # Backward pass
            loss.backward()
            
            # Debug information
            print("Model debugging information:")
            
            # Check parameter gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    
                    print(f"  {name}:")
                    print(f"    Parameter norm: {param_norm:.6f}")
                    print(f"    Gradient norm: {grad_norm:.6f}")
                    
                    # Check for issues
                    if torch.isnan(param.grad).any():
                        print(f"    WARNING: NaN gradients detected!")
                    if torch.isinf(param.grad).any():
                        print(f"    WARNING: Inf gradients detected!")
                    if grad_norm > 100.0:
                        print(f"    WARNING: Large gradients detected!")
                    if grad_norm < 1e-8:
                        print(f"    WARNING: Very small gradients detected!")
            
            # Disable anomaly detection
            torch.autograd.set_detect_anomaly(False)
            
        except Exception as e:
            print(f"  Error during debugging: {e}")
            torch.autograd.set_detect_anomaly(False)
        
        print()


def run_all_advanced_examples():
    """Run all advanced autograd examples."""
    print("Advanced PyTorch Autograd Examples")
    print("=" * 50)
    
    examples = [
        AdvancedAutogradExamples.demonstrate_gradient_flow_analysis,
        AdvancedAutogradExamples.demonstrate_custom_gradient_functions,
        AdvancedAutogradExamples.demonstrate_higher_order_optimization,
        AdvancedAutogradExamples.demonstrate_gradient_based_analysis,
        AdvancedAutogradExamples.demonstrate_autograd_with_custom_models,
        AdvancedAutogradExamples.demonstrate_performance_optimization,
        AdvancedAutogradExamples.demonstrate_gradient_visualization,
        AdvancedAutogradExamples.demonstrate_autograd_debugging
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error running example {example.__name__}: {e}")
            print()
    
    print("Advanced autograd examples completed!")


if __name__ == "__main__":
    run_all_advanced_examples()



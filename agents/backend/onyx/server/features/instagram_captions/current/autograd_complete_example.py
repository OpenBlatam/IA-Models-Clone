"""
Complete PyTorch Autograd System Example

This file demonstrates the complete PyTorch autograd system with all features:
1. Basic autograd operations
2. Custom loss functions with autograd
3. Gradient monitoring and analysis
4. Custom autograd functions
5. Integration with custom model architectures
6. Advanced autograd techniques
7. Performance optimization
8. Debugging and troubleshooting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad, Variable, Function
import numpy as np
import yaml
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt

# Import custom model architectures
try:
    from custom_model_architectures import (
        CustomTransformerModel, CustomCNNModel, CustomRNNModel, 
        CNNTransformerHybrid, create_model_from_config
    )
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: custom_model_architectures not found. Some examples may not work.")
    CUSTOM_MODELS_AVAILABLE = False

# Import autograd system
try:
    from pytorch_autograd_system import (
        AutogradUtils, CustomLossFunction, GradientMonitor, 
        CustomAutogradFunction, AutogradTrainingSystem
    )
    AUTOGRAD_SYSTEM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_autograd_system not found. Some examples may not work.")
    AUTOGRAD_SYSTEM_AVAILABLE = False


class CompleteAutogradExample:
    """Complete example demonstrating all autograd system features."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.setup_components()
    
    def load_config(self) -> Dict:
        """Load autograd system configuration."""
        try:
            with open('autograd_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            print("Configuration loaded successfully")
            return config
        except FileNotFoundError:
            print("Configuration file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration if file not found."""
        return {
            'autograd_system': {
                'enable_gradients': True,
                'gradient_monitoring': {'enabled': True},
                'advanced_features': {'higher_order_gradients': True}
            },
            'training_configs': {
                'basic_training': {
                    'optimizer': 'adam',
                    'learning_rate': 0.001
                }
            }
        }
    
    def setup_components(self):
        """Setup all system components."""
        print("\n=== Setting up Autograd System Components ===")
        
        # Create simple models for demonstration
        self.simple_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(self.device)
        
        # Create custom models if available
        if CUSTOM_MODELS_AVAILABLE:
            self.transformer_model = CustomTransformerModel(
                vocab_size=100, d_model=64, nhead=4, num_layers=2
            ).to(self.device)
            
            self.cnn_model = CustomCNNModel(
                input_channels=3, num_classes=5, base_channels=16
            ).to(self.device)
            
            self.rnn_model = CustomRNNModel(
                input_size=50, hidden_size=64, num_layers=2, num_classes=3
            ).to(self.device)
        
        # Setup optimizers
        self.optimizers = {
            'simple': optim.Adam(self.simple_model.parameters(), lr=0.001),
            'transformer': optim.Adam(self.transformer_model.parameters(), lr=0.0001) if CUSTOM_MODELS_AVAILABLE else None,
            'cnn': optim.Adam(self.cnn_model.parameters(), lr=0.001) if CUSTOM_MODELS_AVAILABLE else None,
            'rnn': optim.Adam(self.rnn_model.parameters(), lr=0.001) if CUSTOM_MODELS_AVAILABLE else None
        }
        
        # Setup loss functions
        self.loss_functions = {
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'custom': CustomLossFunction(alpha=1.0, beta=0.1) if AUTOGRAD_SYSTEM_AVAILABLE else None
        }
        
        print("Components setup completed!")
    
    def demonstrate_basic_autograd(self):
        """Demonstrate basic PyTorch autograd functionality."""
        print("\n=== Basic Autograd Demonstration ===")
        
        # Create tensors with gradients enabled
        x = torch.randn(5, 10, requires_grad=True, device=self.device)
        y = torch.randn(5, 1, device=self.device)
        
        print(f"Input tensor x shape: {x.shape}")
        print(f"Target tensor y shape: {y.shape}")
        
        # Forward pass
        predictions = self.simple_model(x)
        loss = self.loss_functions['mse'](predictions, y)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Loss value: {loss.item():.6f}")
        
        # Backward pass (autograd computes gradients)
        loss.backward()
        
        # Check gradients
        print("\nGradient analysis:")
        for name, param in self.simple_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}")
        
        # Zero gradients for next iteration
        self.simple_model.zero_grad()
        print("Basic autograd demonstration completed!")
    
    def demonstrate_custom_loss_function(self):
        """Demonstrate custom loss function with autograd."""
        if not AUTOGRAD_SYSTEM_AVAILABLE:
            print("Custom loss function not available, skipping...")
            return
        
        print("\n=== Custom Loss Function Demonstration ===")
        
        # Create data
        x = torch.randn(8, 10, requires_grad=True, device=self.device)
        y = torch.randn(8, 1, device=self.device)
        
        # Use custom loss function
        custom_loss_fn = self.loss_functions['custom']
        predictions = self.simple_model(x)
        loss = custom_loss_fn(predictions, y)
        
        print(f"Custom loss value: {loss.item():.6f}")
        print(f"Loss function parameters:")
        print(f"  alpha: {custom_loss_fn.alpha.item():.6f}")
        print(f"  beta: {custom_loss_fn.beta.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients for both model and loss function parameters
        print("\nGradients:")
        print("Model parameters:")
        for name, param in self.simple_model.named_parameters():
            if param.grad is not None:
                print(f"  {name}: {param.grad.norm().item():.6f}")
        
        print("Loss function parameters:")
        print(f"  alpha grad: {custom_loss_fn.alpha.grad.item():.6f}")
        print(f"  beta grad: {custom_loss_fn.beta.grad.item():.6f}")
        
        # Zero gradients
        self.simple_model.zero_grad()
        custom_loss_fn.zero_grad()
        print("Custom loss function demonstration completed!")
    
    def demonstrate_gradient_monitoring(self):
        """Demonstrate gradient monitoring capabilities."""
        if not AUTOGRAD_SYSTEM_AVAILABLE:
            print("Gradient monitoring not available, skipping...")
            return
        
        print("\n=== Gradient Monitoring Demonstration ===")
        
        # Setup gradient monitoring
        monitor = GradientMonitor(self.simple_model)
        hooks = monitor.register_hooks()
        
        print("Gradient monitoring hooks registered")
        
        # Run multiple training steps
        for step in range(3):
            # Create data
            x = torch.randn(4, 10, device=self.device)
            y = torch.randn(4, 1, device=self.device)
            
            # Forward pass
            predictions = self.simple_model(x)
            loss = self.loss_functions['mse'](predictions, y)
            
            # Backward pass
            loss.backward()
            
            # Get gradient summary
            summary = monitor.get_gradient_summary()
            print(f"Step {step + 1}: {summary}")
            
            # Zero gradients
            self.simple_model.zero_grad()
        
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
        print("Gradient monitoring demonstration completed!")
    
    def demonstrate_custom_autograd_function(self):
        """Demonstrate custom autograd function."""
        if not AUTOGRAD_SYSTEM_AVAILABLE:
            print("Custom autograd function not available, skipping...")
            return
        
        print("\n=== Custom Autograd Function Demonstration ===")
        
        # Create custom function
        custom_fn = CustomAutogradFunction.apply
        
        # Create input tensors
        input_tensor = torch.randn(3, 2, requires_grad=True, device=self.device)
        weight_tensor = torch.randn(2, 4, requires_grad=True, device=self.device)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Weight tensor shape: {weight_tensor.shape}")
        
        # Forward pass
        output = custom_fn(input_tensor, weight_tensor)
        print(f"Output shape: {output.shape}")
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        print(f"Loss value: {loss.item():.6f}")
        print(f"Input gradients shape: {input_tensor.grad.shape}")
        print(f"Weight gradients shape: {weight_tensor.grad.shape}")
        
        # Check gradient values
        print(f"Input gradient norm: {input_tensor.grad.norm().item():.6f}")
        print(f"Weight gradient norm: {weight_tensor.grad.norm().item():.6f}")
        
        print("Custom autograd function demonstration completed!")
    
    def demonstrate_higher_order_gradients(self):
        """Demonstrate higher-order gradient computation."""
        print("\n=== Higher-Order Gradients Demonstration ===")
        
        # Create a simple function
        def complex_function(x, y):
            return torch.sin(x) * torch.exp(y) + torch.cos(x + y) ** 2
        
        # Create variables
        x = torch.tensor(1.0, requires_grad=True, device=self.device)
        y = torch.tensor(2.0, requires_grad=True, device=self.device)
        
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
        
        print("Higher-order gradients demonstration completed!")
    
    def demonstrate_autograd_with_custom_models(self):
        """Demonstrate autograd with custom model architectures."""
        if not CUSTOM_MODELS_AVAILABLE:
            print("Custom models not available, skipping...")
            return
        
        print("\n=== Autograd with Custom Models Demonstration ===")
        
        models = {
            'Transformer': self.transformer_model,
            'CNN': self.cnn_model,
            'RNN': self.rnn_model
        }
        
        # Test each model
        for name, model in models.items():
            print(f"\nTesting {name} model:")
            
            try:
                # Create appropriate sample data
                if name == 'Transformer':
                    x = torch.randint(0, 100, (4, 8), device=self.device)  # (batch, seq_len)
                    y = torch.randint(0, 100, (4, 8), device=self.device)
                elif name == 'CNN':
                    x = torch.randn(4, 3, 32, 32, device=self.device)  # (batch, channels, height, width)
                    y = torch.randint(0, 5, (4,), device=self.device)
                else:  # RNN
                    x = torch.randn(4, 8, 50, device=self.device)  # (batch, seq_len, input_size)
                    y = torch.randint(0, 3, (4,), device=self.device)
                
                # Forward pass
                predictions = model(x)
                
                # Compute appropriate loss
                if name == 'CNN' or name == 'RNN':
                    loss = self.loss_functions['cross_entropy'](predictions, y)
                else:
                    loss = self.loss_functions['cross_entropy'](
                        predictions.view(-1, predictions.size(-1)), 
                        y.view(-1)
                    )
                
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
        
        print("Custom models autograd demonstration completed!")
    
    def demonstrate_performance_optimization(self):
        """Demonstrate autograd performance optimization techniques."""
        print("\n=== Performance Optimization Demonstration ===")
        
        # Create a larger model for testing
        large_model = nn.Sequential(
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        ).to(self.device)
        
        # Create data
        x = torch.randn(32, 500, device=self.device)
        y = torch.randn(32, 100, device=self.device)
        
        # Benchmark different approaches
        print("Benchmarking different approaches:")
        
        # Standard approach
        start_time = time.time()
        for _ in range(5):
            large_model.zero_grad()
            predictions = large_model(x)
            loss = F.mse_loss(predictions, y)
            loss.backward()
        
        standard_time = time.time() - start_time
        print(f"  Standard approach: {standard_time:.4f}s")
        
        # With gradient checkpointing (if available)
        if hasattr(large_model, 'gradient_checkpointing'):
            large_model.gradient_checkpointing = True
            start_time = time.time()
            for _ in range(5):
                large_model.zero_grad()
                predictions = large_model(x)
                loss = F.mse_loss(predictions, y)
                loss.backward()
            
            checkpoint_time = time.time() - start_time
            print(f"  With gradient checkpointing: {checkpoint_time:.4f}s")
            print(f"  Memory efficiency: {standard_time/checkpoint_time:.2f}x")
        
        print("Performance optimization demonstration completed!")
    
    def demonstrate_gradient_debugging(self):
        """Demonstrate autograd debugging techniques."""
        print("\n=== Gradient Debugging Demonstration ===")
        
        # Create a model with potential issues
        debug_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(self.device)
        
        # Create data
        x = torch.randn(16, 10, device=self.device)
        y = torch.randn(16, 1, device=self.device)
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        try:
            # Forward pass
            predictions = debug_model(x)
            loss = F.mse_loss(predictions, y)
            
            # Backward pass
            loss.backward()
            
            # Debug information
            print("Model debugging information:")
            
            # Check parameter gradients
            for name, param in debug_model.named_parameters():
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
        
        print("Gradient debugging demonstration completed!")
    
    def demonstrate_complete_training_loop(self):
        """Demonstrate complete training loop with autograd."""
        print("\n=== Complete Training Loop Demonstration ===")
        
        # Create training data
        num_samples = 100
        x_train = torch.randn(num_samples, 10, device=self.device)
        y_train = torch.randn(num_samples, 1, device=self.device)
        
        # Training loop
        num_epochs = 3
        batch_size = 16
        
        print(f"Training for {num_epochs} epochs with batch size {batch_size}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Process batches
            for i in range(0, num_samples, batch_size):
                batch_x = x_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                predictions = self.simple_model(batch_x)
                loss = self.loss_functions['mse'](predictions, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizers['simple'].step()
                
                # Zero gradients
                self.simple_model.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}")
        
        print("Complete training loop demonstration completed!")
    
    def run_all_demonstrations(self):
        """Run all autograd demonstrations."""
        print("Complete PyTorch Autograd System Demonstration")
        print("=" * 60)
        
        demonstrations = [
            self.demonstrate_basic_autograd,
            self.demonstrate_custom_loss_function,
            self.demonstrate_gradient_monitoring,
            self.demonstrate_custom_autograd_function,
            self.demonstrate_higher_order_gradients,
            self.demonstrate_autograd_with_custom_models,
            self.demonstrate_performance_optimization,
            self.demonstrate_gradient_debugging,
            self.demonstrate_complete_training_loop
        ]
        
        for i, demonstration in enumerate(demonstrations, 1):
            try:
                print(f"\n[{i}/{len(demonstrations)}] Running: {demonstration.__name__}")
                demonstration()
            except Exception as e:
                print(f"Error in {demonstration.__name__}: {e}")
                print("Continuing with next demonstration...")
        
        print("\n" + "=" * 60)
        print("All autograd demonstrations completed!")
        print("The PyTorch autograd system is now fully demonstrated.")


def main():
    """Main function to run the complete autograd example."""
    try:
        # Create and run the complete example
        example = CompleteAutogradExample()
        example.run_all_demonstrations()
        
    except Exception as e:
        print(f"Error running complete autograd example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



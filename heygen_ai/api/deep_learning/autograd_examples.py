from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable, grad
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
PyTorch Autograd Examples for HeyGen AI.

Comprehensive examples demonstrating PyTorch autograd for automatic
differentiation, custom gradients, and advanced training techniques
following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class CustomAutogradExample:
    """Examples of custom autograd functions."""

    @staticmethod
    def custom_activation_function():
        """Example of custom activation function with autograd."""
        
        class CustomReLU(Function):
            """Custom ReLU activation with autograd support."""
            
            @staticmethod
            def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
                """Forward pass of custom ReLU.
                
                Args:
                    ctx: Context for storing tensors.
                    input_tensor: Input tensor.
                    
                Returns:
                    torch.Tensor: Output tensor.
                """
                ctx.save_for_backward(input_tensor)
                return torch.clamp(input_tensor, min=0.0)
            
            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                """Backward pass of custom ReLU.
                
                Args:
                    ctx: Context containing saved tensors.
                    grad_output: Gradient of output.
                    
                Returns:
                    torch.Tensor: Gradient with respect to input.
                """
                input_tensor, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input_tensor < 0] = 0
                return grad_input
        
        # Usage example
        custom_relu = CustomReLU.apply
        
        # Test the function
        x = torch.randn(5, requires_grad=True)
        y = custom_relu(x)
        y.backward(torch.ones_like(y))
        
        logger.info(f"Custom ReLU - Input: {x}")
        logger.info(f"Custom ReLU - Output: {y}")
        logger.info(f"Custom ReLU - Gradient: {x.grad}")
        
        return custom_relu

    @staticmethod
    def custom_loss_function():
        """Example of custom loss function with autograd."""
        
        class CustomHuberLoss(Function):
            """Custom Huber loss with autograd support."""
            
            @staticmethod
            def forward(ctx, predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
                """Forward pass of custom Huber loss.
                
                Args:
                    ctx: Context for storing tensors.
                    predictions: Model predictions.
                    targets: Target values.
                    delta: Huber loss parameter.
                    
                Returns:
                    torch.Tensor: Computed loss.
                """
                ctx.save_for_backward(predictions, targets)
                ctx.delta = delta
                
                diff = predictions - targets
                abs_diff = torch.abs(diff)
                quadratic = torch.clamp(abs_diff, max=delta)
                linear = abs_diff - quadratic
                loss = 0.5 * quadratic.pow(2) + delta * linear
                
                return loss.mean()
            
            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                """Backward pass of custom Huber loss.
                
                Args:
                    ctx: Context containing saved tensors.
                    grad_output: Gradient of output.
                    
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to inputs.
                """
                predictions, targets = ctx.saved_tensors
                delta = ctx.delta
                
                diff = predictions - targets
                abs_diff = torch.abs(diff)
                
                # Gradient computation
                grad_pred = torch.where(abs_diff <= delta, diff, delta * torch.sign(diff))
                grad_pred = grad_pred * grad_output / predictions.numel()
                
                grad_target = -grad_pred
                
                return grad_pred, grad_target
        
        # Usage example
        custom_huber_loss = CustomHuberLoss.apply
        
        # Test the function
        predictions = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)
        
        loss = custom_huber_loss(predictions, targets, delta=1.0)
        loss.backward()
        
        logger.info(f"Custom Huber Loss - Loss: {loss.item():.4f}")
        logger.info(f"Custom Huber Loss - Gradient: {predictions.grad}")
        
        return custom_huber_loss

    @staticmethod
    def custom_layer_function():
        """Example of custom layer with autograd."""
        
        class CustomLinear(Function):
            """Custom linear layer with autograd support."""
            
            @staticmethod
            def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
                """Forward pass of custom linear layer.
                
                Args:
                    ctx: Context for storing tensors.
                    input_tensor: Input tensor.
                    weight: Weight matrix.
                    bias: Optional bias vector.
                    
                Returns:
                    torch.Tensor: Output tensor.
                """
                ctx.save_for_backward(input_tensor, weight, bias)
                
                output = torch.matmul(input_tensor, weight.t())
                if bias is not None:
                    output += bias
                
                return output
            
            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                """Backward pass of custom linear layer.
                
                Args:
                    ctx: Context containing saved tensors.
                    grad_output: Gradient of output.
                    
                Returns:
                    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Gradients with respect to inputs.
                """
                input_tensor, weight, bias = ctx.saved_tensors
                
                grad_input = torch.matmul(grad_output, weight)
                grad_weight = torch.matmul(grad_output.t(), input_tensor)
                
                if bias is not None:
                    grad_bias = grad_output.sum(dim=0)
                else:
                    grad_bias = None
                
                return grad_input, grad_weight, grad_bias
        
        # Usage example
        custom_linear = CustomLinear.apply
        
        # Test the function
        input_tensor = torch.randn(5, 10, requires_grad=True)
        weight = torch.randn(3, 10, requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        
        output = custom_linear(input_tensor, weight, bias)
        output.backward(torch.ones_like(output))
        
        logger.info(f"Custom Linear - Output shape: {output.shape}")
        logger.info(f"Custom Linear - Input gradient shape: {input_tensor.grad.shape}")
        logger.info(f"Custom Linear - Weight gradient shape: {weight.grad.shape}")
        
        return custom_linear


class AutogradTrainingExamples:
    """Examples of autograd usage in training."""

    @staticmethod
    def basic_training_example():
        """Basic training example with autograd."""
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(100):
            # Generate dummy data
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            
            # Backward pass (autograd computes gradients)
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return model

    @staticmethod
    def gradient_accumulation_example():
        """Example of gradient accumulation with autograd."""
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Gradient accumulation parameters
        accumulation_steps = 4
        effective_batch_size = 32
        
        # Training loop with gradient accumulation
        for epoch in range(100):
            optimizer.zero_grad()
            
            for step in range(accumulation_steps):
                # Generate mini-batch
                x = torch.randn(effective_batch_size // accumulation_steps, 10)
                y = torch.randn(effective_batch_size // accumulation_steps, 1)
                
                # Forward pass
                output = model(x)
                loss = F.mse_loss(output, y)
                
                # Scale loss for accumulation
                loss = loss / accumulation_steps
                
                # Backward pass (gradients are accumulated)
                loss.backward()
            
            # Update parameters after accumulation
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item() * accumulation_steps:.4f}")
        
        return model

    @staticmethod
    def custom_gradient_example():
        """Example of custom gradient computation."""
        
        class CustomGradientModel(nn.Module):
            """Model with custom gradient computation."""
            
            def __init__(self) -> Any:
                super().__init__()
                self.linear = nn.Linear(10, 1)
                self.custom_weight = nn.Parameter(torch.randn(1))
            
            def forward(self, x) -> Any:
                # Standard forward pass
                output = self.linear(x)
                
                # Custom computation with manual gradient
                custom_output = output * self.custom_weight
                
                return custom_output
            
            def custom_backward(self, grad_output) -> Any:
                """Custom backward pass for demonstration."""
                
                # Manual gradient computation
                with torch.enable_grad():
                    x = torch.randn(5, 10, requires_grad=True)
                    output = self.forward(x)
                    
                    # Compute gradients manually
                    grad_x = grad(output, x, grad_outputs=torch.ones_like(output))[0]
                    
                    return grad_x
        
        # Create model
        model = CustomGradientModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(50):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            
            # Standard autograd backward pass
            loss.backward()
            
            # Custom gradient computation (for demonstration)
            custom_grad = model.custom_backward(torch.ones_like(output))
            
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                logger.info(f"Custom gradient norm: {custom_grad.norm():.4f}")
        
        return model


class AutogradAdvancedExamples:
    """Advanced autograd examples."""

    @staticmethod
    def second_order_gradients_example():
        """Example of second-order gradients with autograd."""
        
        # Create model
        model = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        
        # Create data
        x = torch.randn(10, 5, requires_grad=True)
        y = torch.randn(10, 1)
        
        # First forward pass
        output = model(x)
        loss = F.mse_loss(output, y)
        
        # First backward pass (create_graph=True for second-order gradients)
        loss.backward(create_graph=True)
        
        # Compute second-order gradients
        second_order_grads = []
        for param in model.parameters():
            if param.grad is not None:
                # Compute gradient of gradient
                grad_grad = grad(param.grad.sum(), x, retain_graph=True)[0]
                second_order_grads.append(grad_grad)
        
        logger.info(f"Second-order gradients computed for {len(second_order_grads)} parameters")
        
        return second_order_grads

    @staticmethod
    def gradient_checkpointing_example():
        """Example of gradient checkpointing with autograd."""
        
        class LargeModel(nn.Module):
            """Large model for demonstrating gradient checkpointing."""
            
            def __init__(self) -> Any:
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(100, 100) for _ in range(20)
                ])
                self.activation = nn.ReLU()
            
            def forward(self, x) -> Any:
                for layer in self.layers:
                    # Use gradient checkpointing for memory efficiency
                    x = torch.utils.checkpoint.checkpoint(
                        self._forward_layer, layer, x
                    )
                return x
            
            def _forward_layer(self, layer, x) -> Any:
                return self.activation(layer(x))
        
        # Create model
        model = LargeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(10):
            x = torch.randn(32, 100)
            y = torch.randn(32, 100)
            
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            
            # Backward pass with gradient checkpointing
            loss.backward()
            optimizer.step()
            
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return model

    @staticmethod
    def custom_autograd_function_example():
        """Example of complex custom autograd function."""
        
        class ComplexCustomFunction(Function):
            """Complex custom function with autograd support."""
            
            @staticmethod
            def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
                """Forward pass with complex computation."""
                ctx.save_for_backward(input_tensor, weight, bias)
                
                # Complex forward computation
                linear_output = torch.matmul(input_tensor, weight.t()) + bias
                activated_output = torch.relu(linear_output)
                normalized_output = F.layer_norm(activated_output, activated_output.shape[1:])
                
                return normalized_output
            
            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Backward pass with complex gradient computation."""
                input_tensor, weight, bias = ctx.saved_tensors
                
                # Recompute forward pass for gradients
                linear_output = torch.matmul(input_tensor, weight.t()) + bias
                activated_output = torch.relu(linear_output)
                
                # Compute gradients through the computation graph
                grad_activated = grad_output
                grad_linear = grad_activated * (activated_output > 0).float()
                grad_input = torch.matmul(grad_linear, weight)
                grad_weight = torch.matmul(grad_linear.t(), input_tensor)
                grad_bias = grad_linear.sum(dim=0)
                
                return grad_input, grad_weight, grad_bias
        
        # Usage example
        custom_function = ComplexCustomFunction.apply
        
        # Test the function
        input_tensor = torch.randn(5, 10, requires_grad=True)
        weight = torch.randn(5, 10, requires_grad=True)
        bias = torch.randn(5, requires_grad=True)
        
        output = custom_function(input_tensor, weight, bias)
        output.backward(torch.ones_like(output))
        
        logger.info(f"Complex Custom Function - Output shape: {output.shape}")
        logger.info(f"Complex Custom Function - Input gradient: {input_tensor.grad.norm():.4f}")
        logger.info(f"Complex Custom Function - Weight gradient: {weight.grad.norm():.4f}")
        
        return custom_function


def run_autograd_examples():
    """Run all autograd examples."""
    
    logger.info("Running PyTorch Autograd Examples")
    logger.info("=" * 50)
    
    # Custom autograd examples
    logger.info("\n1. Custom Autograd Functions:")
    custom_relu = CustomAutogradExample.custom_activation_function()
    custom_huber_loss = CustomAutogradExample.custom_loss_function()
    custom_linear = CustomAutogradExample.custom_layer_function()
    
    # Training examples
    logger.info("\n2. Training Examples:")
    basic_model = AutogradTrainingExamples.basic_training_example()
    accumulation_model = AutogradTrainingExamples.gradient_accumulation_example()
    custom_gradient_model = AutogradTrainingExamples.custom_gradient_example()
    
    # Advanced examples
    logger.info("\n3. Advanced Examples:")
    second_order_grads = AutogradAdvancedExamples.second_order_gradients_example()
    checkpointing_model = AutogradAdvancedExamples.gradient_checkpointing_example()
    complex_function = AutogradAdvancedExamples.custom_autograd_function_example()
    
    logger.info("\nAll autograd examples completed successfully!")
    
    return {
        "custom_functions": {
            "relu": custom_relu,
            "huber_loss": custom_huber_loss,
            "linear": custom_linear
        },
        "training_models": {
            "basic": basic_model,
            "accumulation": accumulation_model,
            "custom_gradient": custom_gradient_model
        },
        "advanced_examples": {
            "second_order_grads": second_order_grads,
            "checkpointing_model": checkpointing_model,
            "complex_function": complex_function
        }
    }


if __name__ == "__main__":
    # Run examples
    examples = run_autograd_examples()
    logger.info("PyTorch Autograd Examples completed!") 
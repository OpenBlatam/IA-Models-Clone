#!/usr/bin/env python3
"""
Advanced PyTorch Autograd Usage for Blaze AI
Demonstrates custom backward functions, gradient manipulation, and autograd best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable, grad
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AutogradConfig:
    """Configuration for autograd operations"""
    enable_grad: bool = True
    create_graph: bool = False
    retain_graph: bool = False
    allow_unused: bool = False
    use_double_backward: bool = False
    gradient_clipping: float = 1.0
    custom_gradient_scale: float = 1.0


class CustomBackwardFunction(Function):
    """Custom backward function with gradient manipulation"""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Forward pass with custom computation"""
        ctx.save_for_backward(input_tensor)
        ctx.scale_factor = scale_factor
        
        # Custom forward computation
        output = input_tensor * scale_factor + torch.sin(input_tensor)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Custom backward pass with gradient manipulation"""
        input_tensor, = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        
        # Custom gradient computation
        grad_input = grad_output * (scale_factor + torch.cos(input_tensor))
        
        # Scale gradients for demonstration
        grad_input = grad_input * ctx.scale_factor
        
        return grad_input, None


class GradientReversalFunction(Function):
    """Gradient reversal for adversarial training"""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return input_tensor
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Reverse gradient direction
        return grad_output.neg() * ctx.alpha


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for domain adaptation"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class CustomLossFunction(nn.Module):
    """Custom loss function with autograd"""
    
    def __init__(self, config: AutogradConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Custom loss computation with gradient manipulation"""
        
        # Ensure gradients are computed
        if not predictions.requires_grad:
            predictions.requires_grad_(True)
        
        # Custom loss computation
        loss = F.mse_loss(predictions, targets)
        
        # Add custom regularization term
        if predictions.requires_grad:
            # Compute gradients with respect to predictions
            grad_pred = grad(loss, predictions, create_graph=True, retain_graph=True)[0]
            
            # Apply gradient scaling
            grad_pred = grad_pred * self.config.custom_gradient_scale
            
            # Add gradient penalty
            grad_norm = torch.norm(grad_pred, p=2)
            loss = loss + 0.1 * grad_norm
        
        return loss


class AutogradTrainer:
    """Trainer with advanced autograd usage"""
    
    def __init__(self, model: nn.Module, config: AutogradConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Custom loss function
        self.criterion = CustomLossFunction(config)
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    def compute_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradients with advanced autograd features"""
        
        # Enable gradient computation
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        loss = self.criterion(outputs, targets)
        
        # Backward pass with custom options
        if self.config.create_graph:
            # Create computational graph for higher-order gradients
            loss.backward(create_graph=True, retain_graph=True)
        else:
            loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        
        return {
            'loss': loss,
            'gradients': gradients,
            'outputs': outputs
        }
    
    def compute_second_order_gradients(self, inputs: torch.Tensor, 
                                     targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute second-order gradients using autograd"""
        
        # First forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # First backward pass
        first_grads = grad(loss, self.model.parameters(), create_graph=True, retain_graph=True)
        
        # Second backward pass for Hessian approximation
        second_grads = []
        for param, first_grad in zip(self.model.parameters(), first_grads):
            if first_grad is not None:
                # Compute second derivative
                second_grad = grad(first_grad, param, create_graph=False, retain_graph=True)[0]
                second_grads.append(second_grad)
            else:
                second_grads.append(None)
        
        return {
            'loss': loss,
            'first_gradients': first_grads,
            'second_gradients': second_grads,
            'outputs': outputs
        }
    
    def gradient_analysis(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze gradient statistics"""
        
        grad_norms = {}
        grad_means = {}
        grad_stds = {}
        
        for name, grad_tensor in gradients.items():
            if grad_tensor is not None:
                grad_norms[name] = torch.norm(grad_tensor, p=2).item()
                grad_means[name] = torch.mean(grad_tensor).item()
                grad_stds[name] = torch.std(grad_tensor).item()
        
        return {
            'gradient_norms': grad_norms,
            'gradient_means': grad_means,
            'gradient_stds': grad_stds
        }


class AutogradMonitor:
    """Monitor autograd operations and gradients"""
    
    def __init__(self):
        self.gradient_history = []
        self.computation_graphs = []
    
    def monitor_gradients(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """Monitor gradient statistics during training"""
        
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'norm': torch.norm(param.grad, p=2).item(),
                    'mean': torch.mean(param.grad).item(),
                    'std': torch.std(param.grad).item(),
                    'max': torch.max(param.grad).item(),
                    'min': torch.min(param.grad).item()
                }
        
        self.gradient_history.append({
            'step': step,
            'gradients': grad_stats
        })
        
        return grad_stats
    
    def detect_gradient_issues(self, grad_stats: Dict[str, Any]) -> List[str]:
        """Detect potential gradient issues"""
        
        issues = []
        
        for name, stats in grad_stats.items():
            # Check for vanishing gradients
            if stats['norm'] < 1e-6:
                issues.append(f"Vanishing gradients in {name}")
            
            # Check for exploding gradients
            if stats['norm'] > 10.0:
                issues.append(f"Exploding gradients in {name}")
            
            # Check for NaN gradients
            if torch.isnan(torch.tensor(stats['norm'])):
                issues.append(f"NaN gradients in {name}")
        
        return issues


class CustomOptimizer(nn.Module):
    """Custom optimizer using autograd for meta-learning"""
    
    def __init__(self, base_optimizer: torch.optim.Optimizer, meta_lr: float = 0.01):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.meta_lr = meta_lr
    
    def meta_update(self, model: nn.Module, loss_fn, 
                   support_data: Tuple[torch.Tensor, torch.Tensor],
                   query_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Meta-learning update using autograd"""
        
        # Clone model for inner loop
        inner_model = type(model)(*model.parameters())
        inner_model.load_state_dict(model.state_dict())
        
        # Inner loop optimization
        inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.01)
        
        for _ in range(5):  # 5 inner steps
            support_inputs, support_targets = support_data
            support_outputs = inner_model(support_inputs)
            inner_loss = loss_fn(support_outputs, support_targets)
            
            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()
        
        # Outer loop: compute loss on query set
        query_inputs, query_targets = query_data
        query_outputs = inner_model(query_inputs)
        query_loss = loss_fn(query_outputs, query_targets)
        
        return query_loss


class AutogradExperiments:
    """Collection of autograd experiments and demonstrations"""
    
    @staticmethod
    def demonstrate_gradient_flow():
        """Demonstrate gradient flow through different operations"""
        
        logger.info("Demonstrating gradient flow...")
        
        # Create input tensor
        x = torch.randn(3, 3, requires_grad=True)
        logger.info(f"Input tensor: {x}")
        logger.info(f"Requires grad: {x.requires_grad}")
        
        # Apply custom backward function
        y = CustomBackwardFunction.apply(x, 2.0)
        logger.info(f"Output after custom function: {y}")
        
        # Compute gradients
        y.backward(torch.ones_like(y))
        logger.info(f"Gradients: {x.grad}")
        
        return x.grad
    
    @staticmethod
    def demonstrate_second_order_gradients():
        """Demonstrate second-order gradient computation"""
        
        logger.info("Demonstrating second-order gradients...")
        
        # Create model and input
        model = nn.Linear(5, 1)
        x = torch.randn(10, 5, requires_grad=True)
        y = torch.randn(10, 1)
        
        # Forward pass
        output = model(x)
        loss = F.mse_loss(output, y)
        
        # First-order gradients
        first_grads = grad(loss, model.parameters(), create_graph=True)
        logger.info(f"First-order gradients computed: {len(first_grads)}")
        
        # Second-order gradients
        second_grads = []
        for param, first_grad in zip(model.parameters(), first_grads):
            if first_grad is not None:
                second_grad = grad(first_grad, param, create_graph=False)[0]
                second_grads.append(second_grad)
        
        logger.info(f"Second-order gradients computed: {len(second_grads)}")
        
        return first_grads, second_grads
    
    @staticmethod
    def demonstrate_gradient_reversal():
        """Demonstrate gradient reversal for adversarial training"""
        
        logger.info("Demonstrating gradient reversal...")
        
        # Create model and data
        model = nn.Linear(10, 1)
        x = torch.randn(5, 10, requires_grad=True)
        y = torch.randn(5, 1)
        
        # Apply gradient reversal
        reversed_x = GradientReversalFunction.apply(x, 0.5)
        
        # Forward pass
        output = model(reversed_x)
        loss = F.mse_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        logger.info(f"Original gradients: {x.grad}")
        logger.info(f"Reversed gradients: {reversed_x.grad}")
        
        return x.grad, reversed_x.grad
    
    @staticmethod
    def demonstrate_custom_loss_gradients():
        """Demonstrate custom loss function with gradient manipulation"""
        
        logger.info("Demonstrating custom loss gradients...")
        
        # Create model and data
        model = nn.Linear(5, 1)
        x = torch.randn(10, 5)
        y = torch.randn(10, 1)
        
        # Custom loss function
        config = AutogradConfig(create_graph=True, custom_gradient_scale=2.0)
        criterion = CustomLossFunction(config)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        logger.info(f"Custom loss gradients: {gradients}")
        
        return gradients


def main():
    """Main execution function"""
    logger.info("Starting Advanced PyTorch Autograd Demonstrations...")
    
    # Create configuration
    config = AutogradConfig(
        enable_grad=True,
        create_graph=True,
        retain_graph=True,
        use_double_backward=True,
        gradient_clipping=1.0,
        custom_gradient_scale=2.0
    )
    
    # Demonstrate gradient flow
    grad_flow = AutogradExperiments.demonstrate_gradient_flow()
    logger.info(f"Gradient flow demonstration completed")
    
    # Demonstrate second-order gradients
    first_grads, second_grads = AutogradExperiments.demonstrate_second_order_gradients()
    logger.info(f"Second-order gradients demonstration completed")
    
    # Demonstrate gradient reversal
    orig_grads, rev_grads = AutogradExperiments.demonstrate_gradient_reversal()
    logger.info(f"Gradient reversal demonstration completed")
    
    # Demonstrate custom loss gradients
    custom_grads = AutogradExperiments.demonstrate_custom_loss_gradients()
    logger.info(f"Custom loss gradients demonstration completed")
    
    # Create simple model for training demonstration
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Create trainer
    trainer = AutogradTrainer(model, config)
    
    # Create sample data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # Compute gradients
    results = trainer.compute_gradients(inputs, targets)
    logger.info(f"Training gradients computed: loss={results['loss']:.4f}")
    
    # Analyze gradients
    monitor = AutogradMonitor()
    grad_stats = monitor.monitor_gradients(model, step=0)
    logger.info(f"Gradient statistics: {grad_stats}")
    
    # Detect issues
    issues = monitor.detect_gradient_issues(grad_stats)
    if issues:
        logger.warning(f"Gradient issues detected: {issues}")
    else:
        logger.info("No gradient issues detected")
    
    logger.info("Advanced PyTorch Autograd demonstrations completed successfully!")


if __name__ == "__main__":
    main()

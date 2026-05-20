"""
PyTorch Autograd Automatic Differentiation System
Comprehensive implementation demonstrating PyTorch's autograd capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AutogradConfig:
    """Configuration for autograd demonstrations."""
    
    # Model parameters
    input_size: int = 10
    hidden_size: int = 20
    output_size: int = 5
    batch_size: int = 32
    
    # Training parameters
    learning_rate: float = 0.01
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # Autograd specific
    enable_grad: bool = True
    create_graph: bool = False
    retain_graph: bool = False
    allow_unused: bool = False
    
    # Debugging
    detect_anomaly: bool = False
    grad_check_numerics: bool = True


class CustomAutogradFunction(autograd.Function):
    """Custom autograd function demonstrating manual gradient computation."""
    
    @staticmethod
    def forward(ctx, input_tensor, weight, bias):
        """Forward pass with gradient computation preparation."""
        ctx.save_for_backward(input_tensor, weight, bias)
        
        # Custom forward computation
        output = torch.matmul(input_tensor, weight.t()) + bias
        output = torch.tanh(output)  # Activation function
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with manual gradient computation."""
        input_tensor, weight, bias = ctx.saved_tensors
        
        # Compute gradients manually
        # d/dx tanh(x) = 1 - tanh(x)^2
        tanh_output = torch.tanh(torch.matmul(input_tensor, weight.t()) + bias)
        grad_tanh = grad_output * (1 - tanh_output ** 2)
        
        # Gradients for each parameter
        grad_input = torch.matmul(grad_tanh, weight)
        grad_weight = torch.matmul(grad_tanh.t(), input_tensor)
        grad_bias = grad_tanh.sum(dim=0)
        
        return grad_input, grad_weight, grad_bias


class AutogradAwareModel(nn.Module):
    """Model demonstrating various autograd features."""
    
    def __init__(self, config: AutogradConfig):
        super().__init__()
        self.config = config
        
        # Linear layers
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.output_size)
        
        # Custom autograd function
        self.custom_function = CustomAutogradFunction.apply
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """Forward pass with autograd demonstration."""
        # Standard autograd
        x = self.fc1(x)
        x = F.relu(x)
        
        # Custom autograd function
        x = self.custom_function(x, self.fc2.weight, self.fc2.bias)
        
        return x


class AutogradGradientTracker:
    """Track and analyze gradients during training."""
    
    def __init__(self):
        self.gradient_history = {}
        self.gradient_norms = {}
        self.gradient_means = {}
        self.gradient_stds = {}
    
    def track_gradients(self, model: nn.Module, step: int):
        """Track gradients for all parameters."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.gradient_history:
                    self.gradient_history[name] = []
                    self.gradient_norms[name] = []
                    self.gradient_means[name] = []
                    self.gradient_stds[name] = []
                
                # Store gradient statistics
                grad_data = param.grad.data.clone()
                self.gradient_history[name].append(grad_data)
                self.gradient_norms[name].append(grad_data.norm().item())
                self.gradient_means[name].append(grad_data.mean().item())
                self.gradient_stds[name].append(grad_data.std().item())
    
    def get_gradient_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for gradients."""
        summary = {}
        for name in self.gradient_history.keys():
            if self.gradient_norms[name]:
                summary[name] = {
                    'mean_norm': np.mean(self.gradient_norms[name]),
                    'std_norm': np.std(self.gradient_norms[name]),
                    'max_norm': np.max(self.gradient_norms[name]),
                    'min_norm': np.min(self.gradient_norms[name]),
                    'mean_grad': np.mean(self.gradient_means[name]),
                    'std_grad': np.mean(self.gradient_stds[name])
                }
        return summary


class AutogradDebugger:
    """Debug autograd computations and detect issues."""
    
    def __init__(self, config: AutogradConfig):
        self.config = config
        self.anomaly_detected = False
        self.numerical_issues = []
    
    def enable_debugging(self):
        """Enable autograd debugging features."""
        if self.config.detect_anomaly:
            autograd.detect_anomaly()
            logging.info("Autograd anomaly detection enabled")
        
        if self.config.grad_check_numerics:
            torch.autograd.set_detect_anomaly(True)
            logging.info("Gradient numerical checking enabled")
    
    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Check gradients for numerical issues."""
        issues = {
            'nan_gradients': [],
            'inf_gradients': [],
            'zero_gradients': [],
            'large_gradients': []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Check for NaN gradients
                if torch.isnan(grad).any():
                    issues['nan_gradients'].append(name)
                
                # Check for Inf gradients
                if torch.isinf(grad).any():
                    issues['inf_gradients'].append(name)
                
                # Check for zero gradients
                if grad.norm() == 0:
                    issues['zero_gradients'].append(name)
                
                # Check for very large gradients
                if grad.norm() > 1000:
                    issues['large_gradients'].append(name)
        
        return issues


class AutogradOptimizer:
    """Custom optimizer demonstrating autograd usage."""
    
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
        
        # Initialize velocity for each parameter
        for param in self.params:
            self.velocity[param] = torch.zeros_like(param.data)
    
    def step(self):
        """Perform optimization step using autograd."""
        for param in self.params:
            if param.grad is None:
                continue
            
            # Momentum update
            self.velocity[param] = (
                self.momentum * self.velocity[param] + 
                self.lr * param.grad.data
            )
            
            # Parameter update
            param.data -= self.velocity[param]
    
    def zero_grad(self):
        """Zero gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()


class AutogradDataset(Dataset):
    """Dataset for autograd demonstrations."""
    
    def __init__(self, num_samples: int, input_size: int, output_size: int):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        
        # Generate synthetic data
        self.inputs = torch.randn(num_samples, input_size)
        self.targets = torch.randn(num_samples, output_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class AutogradTrainer:
    """Trainer demonstrating autograd features."""
    
    def __init__(self, model: nn.Module, config: AutogradConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = AutogradOptimizer(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Tracking and debugging
        self.gradient_tracker = AutogradGradientTracker()
        self.debugger = AutogradDebugger(config)
        
        # Enable debugging if configured
        self.debugger.enable_debugging()
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step with autograd demonstration."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Enable gradient computation
        inputs.requires_grad_(self.config.enable_grad)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Backward pass with autograd
        loss.backward(
            create_graph=self.config.create_graph,
            retain_graph=self.config.retain_graph
        )
        
        # Track gradients
        self.gradient_tracker.track_gradients(self.model, 0)
        
        # Check for issues
        issues = self.debugger.check_gradients(self.model)
        if any(issues.values()):
            logging.warning(f"Gradient issues detected: {issues}")
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'gradient_issues': len([v for v in issues.values() if v])
        }
    
    def compute_hessian(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Hessian matrix using autograd."""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # First forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Compute gradients
        gradients = autograd.grad(
            loss, 
            self.model.parameters(), 
            create_graph=True,
            retain_graph=True
        )
        
        # Flatten gradients
        grad_flat = torch.cat([g.flatten() for g in gradients])
        
        # Compute Hessian
        hessian = []
        for i in range(len(grad_flat)):
            hessian_row = autograd.grad(
                grad_flat[i], 
                self.model.parameters(), 
                create_graph=True,
                retain_graph=True
            )
            hessian_row_flat = torch.cat([h.flatten() for h in hessian_row])
            hessian.append(hessian_row_flat)
        
        return torch.stack(hessian)
    
    def compute_jacobian(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrix using autograd."""
        inputs = inputs.to(self.device)
        batch_size = inputs.size(0)
        
        # Forward pass
        outputs = self.model(inputs)
        output_size = outputs.size(1)
        
        # Compute Jacobian
        jacobian = []
        for i in range(output_size):
            grad_output = torch.zeros_like(outputs)
            grad_output[:, i] = 1.0
            
            gradients = autograd.grad(
                outputs, 
                inputs, 
                grad_outputs=grad_output,
                create_graph=True,
                retain_graph=True
            )
            jacobian.append(gradients[0])
        
        return torch.stack(jacobian, dim=1)
    
    def gradient_penalty(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty using autograd."""
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute gradients with respect to inputs
        gradients = autograd.grad(
            outputs.sum(), 
            inputs, 
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty


class AutogradVisualizer:
    """Visualize autograd computations and gradients."""
    
    def __init__(self):
        self.gradient_history = {}
        self.loss_history = []
    
    def plot_gradients(self, gradient_tracker: AutogradGradientTracker):
        """Plot gradient statistics over time."""
        summary = gradient_tracker.get_gradient_summary()
        
        if not summary:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Gradient Statistics Over Time')
        
        # Plot gradient norms
        ax1 = axes[0, 0]
        for name, stats in summary.items():
            ax1.plot(gradient_tracker.gradient_norms[name], label=name)
        ax1.set_title('Gradient Norms')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Norm')
        ax1.legend()
        ax1.grid(True)
        
        # Plot gradient means
        ax2 = axes[0, 1]
        for name, stats in summary.items():
            ax2.plot(gradient_tracker.gradient_means[name], label=name)
        ax2.set_title('Gradient Means')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Mean')
        ax2.legend()
        ax2.grid(True)
        
        # Plot gradient standard deviations
        ax3 = axes[1, 0]
        for name, stats in summary.items():
            ax3.plot(gradient_tracker.gradient_stds[name], label=name)
        ax3.set_title('Gradient Standard Deviations')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Std')
        ax3.legend()
        ax3.grid(True)
        
        # Plot loss history
        ax4 = axes[1, 1]
        if self.loss_history:
            ax4.plot(self.loss_history)
            ax4.set_title('Loss Over Time')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Loss')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_hessian_eigenvalues(self, hessian: torch.Tensor):
        """Plot eigenvalues of Hessian matrix."""
        eigenvalues = torch.linalg.eigvals(hessian).real
        
        plt.figure(figsize=(10, 6))
        plt.hist(eigenvalues.numpy(), bins=50, alpha=0.7)
        plt.title('Hessian Eigenvalue Distribution')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
        print(f"Hessian condition number: {eigenvalues.max() / eigenvalues.min():.2e}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = AutogradConfig(
        input_size=10,
        hidden_size=20,
        output_size=5,
        batch_size=32,
        learning_rate=0.01,
        num_epochs=50,
        detect_anomaly=True,
        grad_check_numerics=True
    )
    
    # Initialize model and trainer
    model = AutogradAwareModel(config)
    trainer = AutogradTrainer(model, config)
    visualizer = AutogradVisualizer()
    
    # Create dataset
    dataset = AutogradDataset(1000, config.input_size, config.output_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop with autograd demonstration
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Training step
            metrics = trainer.train_step((inputs, targets))
            epoch_loss += metrics['loss']
            num_batches += 1
            
            # Store loss for visualization
            visualizer.loss_history.append(metrics['loss'])
            
            # Log progress
            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Gradient Issues: {metrics['gradient_issues']}"
                )
        
        avg_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")
    
    # Demonstrate advanced autograd features
    sample_inputs, sample_targets = next(iter(dataloader))
    
    # Compute Hessian
    logging.info("Computing Hessian matrix...")
    hessian = trainer.compute_hessian(sample_inputs[:5], sample_targets[:5])
    logging.info(f"Hessian shape: {hessian.shape}")
    
    # Compute Jacobian
    logging.info("Computing Jacobian matrix...")
    jacobian = trainer.compute_jacobian(sample_inputs[:5])
    logging.info(f"Jacobian shape: {jacobian.shape}")
    
    # Compute gradient penalty
    logging.info("Computing gradient penalty...")
    penalty = trainer.gradient_penalty(sample_inputs)
    logging.info(f"Gradient penalty: {penalty.item():.4f}")
    
    # Visualize results
    visualizer.plot_gradients(trainer.gradient_tracker)
    visualizer.plot_hessian_eigenvalues(hessian)
    
    # Print gradient summary
    gradient_summary = trainer.gradient_tracker.get_gradient_summary()
    logging.info("Gradient Summary:")
    for name, stats in gradient_summary.items():
        logging.info(f"{name}: {stats}")
    
    logging.info("Autograd demonstration completed successfully!")






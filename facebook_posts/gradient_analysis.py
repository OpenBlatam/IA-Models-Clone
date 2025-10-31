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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path
import json
import math
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Gradient Analysis Module
Advanced gradient analysis and monitoring with PyTorch autograd.
"""



@dataclass
class GradientAnalysisConfig:
    """Configuration for gradient analysis."""
    model_name: str = "gradient_analysis_model"
    input_dimension: int = 100
    hidden_dimension: int = 50
    output_dimension: int = 10
    num_layers: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 100
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    save_gradients: bool = True
    plot_gradients: bool = True
    log_interval: int = 10


class GradientMonitor:
    """Monitor and analyze gradients during training."""
    
    def __init__(self, model: nn.Module, config: GradientAnalysisConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Gradient tracking
        self.gradient_norms = []
        self.gradient_means = []
        self.gradient_stds = []
        self.layer_gradients = {}
        self.parameter_gradients = {}
        
        # Register hooks
        self._register_gradient_hooks()
    
    def _register_gradient_hooks(self) -> Any:
        """Register gradient hooks for all parameters."""
        for name, param in self.model.named_parameters():
            param.register_hook(self._create_gradient_hook(name))
    
    def _create_gradient_hook(self, param_name: str):
        """Create gradient hook for a parameter."""
        def hook(grad) -> Any:
            if grad is not None:
                # Compute gradient statistics
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                # Store statistics
                self.gradient_norms.append(grad_norm)
                self.gradient_means.append(grad_mean)
                self.gradient_stds.append(grad_std)
                
                # Store parameter-specific gradients
                if param_name not in self.parameter_gradients:
                    self.parameter_gradients[param_name] = []
                self.parameter_gradients[param_name].append({
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'step': len(self.gradient_norms)
                })
        
        return hook
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gradient statistics."""
        if not self.gradient_norms:
            return {}
        
        return {
            'overall': {
                'mean_norm': np.mean(self.gradient_norms),
                'std_norm': np.std(self.gradient_norms),
                'max_norm': np.max(self.gradient_norms),
                'min_norm': np.min(self.gradient_norms),
                'mean_mean': np.mean(self.gradient_means),
                'mean_std': np.mean(self.gradient_stds)
            },
            'parameter_specific': self.parameter_gradients,
            'all_norms': self.gradient_norms,
            'all_means': self.gradient_means,
            'all_stds': self.gradient_stds
        }
    
    def plot_gradient_analysis(self, save_path: Optional[str] = None):
        """Plot comprehensive gradient analysis."""
        if not self.gradient_norms:
            print("No gradient data available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        steps = range(len(self.gradient_norms))
        
        # Overall gradient norm
        axes[0, 0].plot(steps, self.gradient_norms, 'b-', alpha=0.7)
        axes[0, 0].set_title('Gradient Norms Over Time')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient mean
        axes[0, 1].plot(steps, self.gradient_means, 'r-', alpha=0.7)
        axes[0, 1].set_title('Gradient Means Over Time')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Gradient Mean')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient std
        axes[0, 2].plot(steps, self.gradient_stds, 'g-', alpha=0.7)
        axes[0, 2].set_title('Gradient Standard Deviations Over Time')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Gradient Std')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Parameter-specific gradients
        if self.parameter_gradients:
            param_names = list(self.parameter_gradients.keys())
            for i, param_name in enumerate(param_names[:3]):  # Plot first 3 parameters
                param_data = self.parameter_gradients[param_name]
                norms = [d['norm'] for d in param_data]
                steps_param = range(len(norms))
                
                axes[1, i].plot(steps_param, norms, alpha=0.7)
                axes[1, i].set_title(f'Gradient Norm: {param_name}')
                axes[1, i].set_xlabel('Training Step')
                axes[1, i].set_ylabel('Gradient Norm')
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('gradient_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.close()


class GradientFlowAnalyzer:
    """Analyze gradient flow through the network."""
    
    def __init__(self, model: nn.Module):
        
    """__init__ function."""
self.model = model
        self.gradient_flow_data = {}
        self.layer_gradients = {}
    
    def analyze_gradient_flow(self, input_tensor: torch.Tensor, 
                            target_tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient flow through the network."""
        # Clear previous data
        self.gradient_flow_data = {}
        self.layer_gradients = {}
        
        # Register hooks for each layer
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM)):
                hook = module.register_forward_hook(self._create_forward_hook(name))
                hooks.append(hook)
        
        # Forward pass
        output = self.model(input_tensor)
        loss = F.mse_loss(output, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze gradient flow
        flow_analysis = self._compute_gradient_flow()
        
        return flow_analysis
    
    def _create_forward_hook(self, layer_name: str):
        """Create forward hook for layer analysis."""
        def hook(module, input_tensor, output_tensor) -> Any:
            self.gradient_flow_data[layer_name] = {
                'input_shape': [t.shape for t in input_tensor],
                'output_shape': output_tensor.shape,
                'input_norm': [t.norm().item() for t in input_tensor],
                'output_norm': output_tensor.norm().item()
            }
        return hook
    
    def _compute_gradient_flow(self) -> Dict[str, Any]:
        """Compute gradient flow statistics."""
        flow_analysis = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                # Determine layer name
                layer_name = '.'.join(name.split('.')[:-1])
                
                if layer_name not in flow_analysis:
                    flow_analysis[layer_name] = {
                        'parameters': [],
                        'total_grad_norm': 0,
                        'avg_grad_norm': 0,
                        'max_grad_norm': 0
                    }
                
                flow_analysis[layer_name]['parameters'].append({
                    'name': name,
                    'grad_norm': grad_norm,
                    'grad_mean': grad_mean,
                    'grad_std': grad_std
                })
                
                flow_analysis[layer_name]['total_grad_norm'] += grad_norm
                flow_analysis[layer_name]['max_grad_norm'] = max(
                    flow_analysis[layer_name]['max_grad_norm'], grad_norm
                )
        
        # Compute averages
        for layer_name in flow_analysis:
            num_params = len(flow_analysis[layer_name]['parameters'])
            flow_analysis[layer_name]['avg_grad_norm'] = (
                flow_analysis[layer_name]['total_grad_norm'] / num_params
            )
        
        return flow_analysis


class CustomGradientFunction(autograd.Function):
    """Custom gradient function for advanced analysis."""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor, 
                bias: torch.Tensor, activation_type: str = "relu") -> torch.Tensor:
        """Forward pass with custom activation."""
        ctx.save_for_backward(input_tensor, weight, bias)
        ctx.activation_type = activation_type
        
        # Linear transformation
        output = torch.matmul(input_tensor, weight.t()) + bias
        
        # Custom activation
        if activation_type == "relu":
            output = F.relu(output)
        elif activation_type == "gelu":
            output = F.gelu(output)
        elif activation_type == "custom":
            output = torch.where(
                output > 0,
                output * torch.exp(-output),
                torch.zeros_like(output)
            )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass with custom gradient computation."""
        input_tensor, weight, bias = ctx.saved_tensors
        activation_type = ctx.activation_type
        
        # Compute activation gradient
        linear_output = torch.matmul(input_tensor, weight.t()) + bias
        
        if activation_type == "relu":
            activation_grad = torch.where(linear_output > 0, torch.ones_like(linear_output), torch.zeros_like(linear_output))
        elif activation_type == "gelu":
            # Approximate GELU gradient
            activation_grad = 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (linear_output + 0.044715 * linear_output**3)))
        elif activation_type == "custom":
            activation_grad = torch.where(
                linear_output > 0,
                (1 - linear_output) * torch.exp(-linear_output),
                torch.zeros_like(linear_output)
            )
        else:
            activation_grad = torch.ones_like(linear_output)
        
        # Apply gradient
        grad_linear = grad_output * activation_grad
        
        # Compute gradients
        grad_input = torch.matmul(grad_linear, weight)
        grad_weight = torch.matmul(grad_linear.t(), input_tensor)
        grad_bias = grad_linear.sum(dim=0)
        
        return grad_input, grad_weight, grad_bias


class AdvancedGradientAnalysisModel(nn.Module):
    """Advanced model for gradient analysis."""
    
    def __init__(self, config: GradientAnalysisConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Layers with different characteristics
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(config.input_dimension, config.hidden_dimension))
        
        # Hidden layers
        for i in range(config.num_layers - 2):
            self.layers.append(nn.Linear(config.hidden_dimension, config.hidden_dimension))
        
        # Output layer
        self.layers.append(nn.Linear(config.hidden_dimension, config.output_dimension))
        
        # Initialize weights with different strategies
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights with different strategies."""
        for i, layer in enumerate(self.layers):
            if i == 0:
                # Xavier initialization for input layer
                nn.init.xavier_uniform_(layer.weight)
            elif i == len(self.layers) - 1:
                # Small weights for output layer
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            else:
                # Orthogonal initialization for hidden layers
                nn.init.orthogonal_(layer.weight)
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient analysis."""
        hidden = input_tensor
        
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                hidden = F.relu(hidden)
        
        return hidden


class GradientAnalysisTrainer:
    """Trainer with comprehensive gradient analysis."""
    
    def __init__(self, model: nn.Module, config: GradientAnalysisConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Gradient monitor
        self.gradient_monitor = GradientMonitor(model, config)
        
        # Gradient flow analyzer
        self.flow_analyzer = GradientFlowAnalyzer(model)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.losses = []
        self.gradient_flow_history = []
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch with gradient analysis."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_data, target_data) in enumerate(dataloader):
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(input_data)
            loss = self.criterion(output, target_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Record loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Analyze gradient flow periodically
            if batch_idx % self.config.log_interval == 0:
                flow_analysis = self.flow_analyzer.analyze_gradient_flow(input_data, target_data)
                self.gradient_flow_history.append(flow_analysis)
        
        return {
            'loss': epoch_loss / num_batches,
            'gradient_stats': self.gradient_monitor.get_gradient_statistics()
        }
    
    def save_analysis_results(self, save_dir: str = "gradient_analysis_results"):
        """Save comprehensive gradient analysis results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save gradient statistics
        gradient_stats = self.gradient_monitor.get_gradient_statistics()
        with open(save_path / "gradient_statistics.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(gradient_stats, f, indent=2, default=str)
        
        # Save gradient flow history
        with open(save_path / "gradient_flow_history.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.gradient_flow_history, f, indent=2, default=str)
        
        # Plot gradient analysis
        self.gradient_monitor.plot_gradient_analysis(save_path / "gradient_analysis.png")
        
        # Save training losses
        with open(save_path / "training_losses.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.losses, f, indent=2)
        
        print(f"Gradient analysis results saved to {save_path}")


def run_gradient_analysis_experiment():
    """Run comprehensive gradient analysis experiment."""
    print("Running Gradient Analysis Experiment")
    print("=" * 50)
    
    # Configuration
    config = GradientAnalysisConfig(
        input_dimension=50,
        hidden_dimension=30,
        output_dimension=10,
        num_layers=4,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=50
    )
    
    # Create model
    model = AdvancedGradientAnalysisModel(config)
    
    # Create trainer
    trainer = GradientAnalysisTrainer(model, config)
    
    # Create dummy dataloader
    class DummyDataset:
        def __init__(self, num_samples=1000, input_dim=50, output_dim=10) -> Any:
            self.input_data = torch.randn(num_samples, input_dim)
            self.target_data = torch.randn(num_samples, output_dim)
        
        def __len__(self) -> Any:
            return len(self.input_data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.input_data[idx], self.target_data[idx]
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(config.num_epochs):
        results = trainer.train_epoch(dataloader)
        trainer.losses.append(results['loss'])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {results['loss']:.6f}")
    
    # Save analysis results
    trainer.save_analysis_results()
    
    print("Gradient analysis experiment completed!")
    return trainer


if __name__ == "__main__":
    # Run gradient analysis experiment
    trainer = run_gradient_analysis_experiment() 
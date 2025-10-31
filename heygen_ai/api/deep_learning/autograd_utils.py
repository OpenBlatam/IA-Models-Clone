from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
PyTorch Autograd Utilities for HeyGen AI.

Advanced utilities for automatic differentiation, gradient computation,
and custom autograd functions following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class CustomAutogradFunction(Function):
    """Custom autograd function for advanced gradient computation."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass of custom autograd function.

        Args:
            ctx: Context for storing tensors for backward pass.
            input_tensor: Input tensor.
            weight: Weight tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ctx.save_for_backward(input_tensor, weight)
        
        # Custom forward computation
        output = torch.matmul(input_tensor, weight)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass of custom autograd function.

        Args:
            ctx: Context containing saved tensors.
            grad_output: Gradient of output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to inputs.
        """
        input_tensor, weight = ctx.saved_tensors
        
        # Compute gradients
        grad_input = torch.matmul(grad_output, weight.t())
        grad_weight = torch.matmul(input_tensor.t(), grad_output)
        
        return grad_input, grad_weight


class GradientComputationUtils:
    """Utilities for advanced gradient computation."""

    def __init__(self) -> Any:
        """Initialize gradient computation utilities."""
        self.gradient_history = []
        self.gradient_norms = []

    def compute_gradients(
        self,
        model: nn.Module,
        loss_function: Callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for model parameters.

        Args:
            model: PyTorch model.
            loss_function: Loss function.
            input_data: Input data.
            target_data: Target data.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of gradients for each parameter.
        """
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model(input_data)
        loss = loss_function(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients

    def compute_gradient_norms(
        self,
        model: nn.Module,
        norm_type: float = 2.0
    ) -> Dict[str, float]:
        """Compute gradient norms for model parameters.

        Args:
            model: PyTorch model.
            norm_type: Type of norm to compute.

        Returns:
            Dict[str, float]: Dictionary of gradient norms for each parameter.
        """
        gradient_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = param.grad.norm(norm_type).item()
        
        return gradient_norms

    def clip_gradients(
        self,
        model: nn.Module,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ) -> float:
        """Clip gradients to prevent exploding gradients.

        Args:
            model: PyTorch model.
            max_norm: Maximum gradient norm.
            norm_type: Type of norm to use.

        Returns:
            float: Total gradient norm before clipping.
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, norm_type
        )
        
        return total_norm.item()

    def compute_hessian(
        self,
        model: nn.Module,
        loss_function: Callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        parameter_name: str
    ) -> torch.Tensor:
        """Compute Hessian matrix for a specific parameter.

        Args:
            model: PyTorch model.
            loss_function: Loss function.
            input_data: Input data.
            target_data: Target data.
            parameter_name: Name of parameter to compute Hessian for.

        Returns:
            torch.Tensor: Hessian matrix.
        """
        # First forward pass
        model.zero_grad()
        output = model(input_data)
        loss = loss_function(output, target_data)
        
        # First backward pass
        loss.backward(create_graph=True)
        
        # Get parameter
        param = dict(model.named_parameters())[parameter_name]
        
        # Compute Hessian
        hessian = []
        for grad in param.grad.flatten():
            model.zero_grad()
            grad.backward(retain_graph=True)
            hessian_row = []
            for p in model.parameters():
                if p.grad is not None:
                    hessian_row.append(p.grad.flatten())
            hessian.append(torch.cat(hessian_row))
        
        return torch.stack(hessian)


class AutogradHooks:
    """Hooks for monitoring autograd computation."""

    def __init__(self) -> Any:
        """Initialize autograd hooks."""
        self.gradient_hooks = []
        self.activation_hooks = []
        self.gradient_stats = {}

    def register_gradient_hooks(self, model: nn.Module):
        """Register hooks to monitor gradients.

        Args:
            model: PyTorch model.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                hook = module.register_backward_hook(
                    self._gradient_hook_fn(name)
                )
                self.gradient_hooks.append(hook)

    def register_activation_hooks(self, model: nn.Module):
        """Register hooks to monitor activations.

        Args:
            model: PyTorch model.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                hook = module.register_forward_hook(
                    self._activation_hook_fn(name)
                )
                self.activation_hooks.append(hook)

    def _gradient_hook_fn(self, name: str):
        """Create gradient hook function.

        Args:
            name: Module name.

        Returns:
            Callable: Gradient hook function.
        """
        def hook_fn(module, grad_input, grad_output) -> Any:
            if grad_output[0] is not None:
                grad_norm = grad_output[0].norm().item()
                if name not in self.gradient_stats:
                    self.gradient_stats[name] = []
                self.gradient_stats[name].append(grad_norm)
        
        return hook_fn

    def _activation_hook_fn(self, name: str):
        """Create activation hook function.

        Args:
            name: Module name.

        Returns:
            Callable: Activation hook function.
        """
        def hook_fn(module, input_tensor, output_tensor) -> Any:
            if name not in self.gradient_stats:
                self.gradient_stats[name] = []
            self.gradient_stats[name].append(output_tensor.mean().item())
        
        return hook_fn

    def remove_hooks(self) -> Any:
        """Remove all registered hooks."""
        for hook in self.gradient_hooks + self.activation_hooks:
            hook.remove()
        self.gradient_hooks = []
        self.activation_hooks = []

    def get_gradient_stats(self) -> Dict[str, List[float]]:
        """Get gradient statistics.

        Returns:
            Dict[str, List[float]]: Gradient statistics.
        """
        return self.gradient_stats.copy()


class CustomLossFunction(nn.Module):
    """Custom loss function with autograd support."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        """Initialize custom loss function.

        Args:
            alpha: Weight for primary loss component.
            beta: Weight for secondary loss component.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of custom loss function.

        Args:
            predictions: Model predictions.
            targets: Target values.
            additional_features: Optional additional features.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Primary loss component
        primary_loss = F.mse_loss(predictions, targets)
        
        # Secondary loss component (if additional features provided)
        if additional_features is not None:
            secondary_loss = F.l1_loss(predictions, additional_features)
            total_loss = self.alpha * primary_loss + self.beta * secondary_loss
        else:
            total_loss = primary_loss
        
        return total_loss


class AutogradOptimizer:
    """Advanced optimizer with autograd monitoring."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = torch.optim.AdamW,
        **optimizer_kwargs
    ):
        """Initialize autograd optimizer.

        Args:
            model: PyTorch model.
            optimizer_class: Optimizer class to use.
            **optimizer_kwargs: Optimizer keyword arguments.
        """
        self.model = model
        self.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        self.gradient_utils = GradientComputationUtils()
        self.autograd_hooks = AutogradHooks()
        
        # Register hooks
        self.autograd_hooks.register_gradient_hooks(model)

    def step(
        self,
        loss_function: Callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        clip_gradients: bool = True,
        max_grad_norm: float = 1.0
    ) -> Dict[str, Any]:
        """Perform optimization step with autograd monitoring.

        Args:
            loss_function: Loss function.
            input_data: Input data.
            target_data: Target data.
            clip_gradients: Whether to clip gradients.
            max_grad_norm: Maximum gradient norm for clipping.

        Returns:
            Dict[str, Any]: Optimization step statistics.
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(input_data)
        loss = loss_function(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient statistics
        gradient_norms = self.gradient_utils.compute_gradient_norms(self.model)
        total_grad_norm = sum(gradient_norms.values())
        
        # Clip gradients if requested
        if clip_gradients:
            clipped_norm = self.gradient_utils.clip_gradients(
                self.model, max_grad_norm
            )
        else:
            clipped_norm = total_grad_norm
        
        # Optimizer step
        self.optimizer.step()
        
        # Get gradient statistics from hooks
        gradient_stats = self.autograd_hooks.get_gradient_stats()
        
        return {
            "loss": loss.item(),
            "total_gradient_norm": total_grad_norm,
            "clipped_gradient_norm": clipped_norm,
            "gradient_norms": gradient_norms,
            "gradient_stats": gradient_stats
        }

    def get_learning_rate(self) -> float:
        """Get current learning rate.

        Returns:
            float: Current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, learning_rate: float):
        """Set learning rate.

        Args:
            learning_rate: New learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate


class AutogradTrainingLoop:
    """Training loop with advanced autograd features."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = torch.optim.AdamW,
        **optimizer_kwargs
    ):
        """Initialize autograd training loop.

        Args:
            model: PyTorch model.
            optimizer_class: Optimizer class to use.
            **optimizer_kwargs: Optimizer keyword arguments.
        """
        self.model = model
        self.autograd_optimizer = AutogradOptimizer(
            model, optimizer_class, **optimizer_kwargs
        )
        self.training_stats = []

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_function: Callable,
        device: torch.device,
        clip_gradients: bool = True,
        max_grad_norm: float = 1.0
    ) -> Dict[str, List[float]]:
        """Train for one epoch with autograd monitoring.

        Args:
            dataloader: Training data loader.
            loss_function: Loss function.
            device: Device to train on.
            clip_gradients: Whether to clip gradients.
            max_grad_norm: Maximum gradient norm for clipping.

        Returns:
            Dict[str, List[float]]: Training statistics.
        """
        self.model.train()
        epoch_stats = {
            "losses": [],
            "gradient_norms": [],
            "learning_rates": []
        }
        
        for batch_idx, (input_data, target_data) in enumerate(dataloader):
            # Move data to device
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            
            # Training step
            step_stats = self.autograd_optimizer.step(
                loss_function=loss_function,
                input_data=input_data,
                target_data=target_data,
                clip_gradients=clip_gradients,
                max_grad_norm=max_grad_norm
            )
            
            # Record statistics
            epoch_stats["losses"].append(step_stats["loss"])
            epoch_stats["gradient_norms"].append(step_stats["clipped_gradient_norm"])
            epoch_stats["learning_rates"].append(
                self.autograd_optimizer.get_learning_rate()
            )
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(
                    f"Batch {batch_idx}: Loss = {step_stats['loss']:.4f}, "
                    f"Grad Norm = {step_stats['clipped_gradient_norm']:.4f}"
                )
        
        return epoch_stats

    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_function: Callable,
        device: torch.device
    ) -> Dict[str, float]:
        """Validate model with autograd monitoring.

        Args:
            dataloader: Validation data loader.
            loss_function: Loss function.
            device: Device to validate on.

        Returns:
            Dict[str, float]: Validation statistics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_data, target_data in dataloader:
                # Move data to device
                input_data = input_data.to(device)
                target_data = target_data.to(device)
                
                # Forward pass
                output = self.model(input_data)
                loss = loss_function(output, target_data)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {
            "validation_loss": avg_loss,
            "num_batches": num_batches
        }


def create_autograd_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    **kwargs
) -> AutogradOptimizer:
    """Factory function to create autograd optimizer.

    Args:
        model: PyTorch model.
        optimizer_type: Type of optimizer.
        **kwargs: Optimizer parameters.

    Returns:
        AutogradOptimizer: Created autograd optimizer.

    Raises:
        ValueError: If optimizer type is not supported.
    """
    optimizer_map = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop
    }
    
    if optimizer_type.lower() not in optimizer_map:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    optimizer_class = optimizer_map[optimizer_type.lower()]
    return AutogradOptimizer(model, optimizer_class, **kwargs)


def create_autograd_training_loop(
    model: nn.Module,
    optimizer_type: str = "adamw",
    **kwargs
) -> AutogradTrainingLoop:
    """Factory function to create autograd training loop.

    Args:
        model: PyTorch model.
        optimizer_type: Type of optimizer.
        **kwargs: Optimizer parameters.

    Returns:
        AutogradTrainingLoop: Created autograd training loop.
    """
    optimizer_map = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop
    }
    
    optimizer_class = optimizer_map.get(optimizer_type.lower(), torch.optim.AdamW)
    return AutogradTrainingLoop(model, optimizer_class, **kwargs) 
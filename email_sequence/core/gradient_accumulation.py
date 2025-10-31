from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from core.training_logger import TrainingLogger, TrainingEventType, LogLevel
from core.error_handling import ErrorHandler, ModelError
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Gradient Accumulation System

Comprehensive gradient accumulation implementation for training with large effective batch sizes
by accumulating gradients over multiple forward/backward passes before updating model parameters.
"""




@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation"""
    
    # Accumulation settings
    accumulation_steps: int = 1  # Number of steps to accumulate gradients
    effective_batch_size: Optional[int] = None  # Target effective batch size
    auto_calculate_steps: bool = True  # Automatically calculate accumulation steps
    
    # Scaling settings
    scale_loss: bool = True  # Scale loss by accumulation steps
    scale_gradients: bool = True  # Scale gradients by accumulation steps
    
    # Memory optimization
    clear_gradients: bool = True  # Clear gradients after accumulation
    memory_efficient: bool = False  # Use memory-efficient accumulation
    
    # Monitoring
    enable_monitoring: bool = True  # Enable accumulation monitoring
    log_accumulation_stats: bool = True  # Log accumulation statistics
    
    # Validation
    validate_accumulation: bool = True  # Validate accumulation correctness
    check_gradient_norms: bool = True  # Check gradient norms during accumulation


class GradientAccumulator:
    """Gradient accumulation manager"""
    
    def __init__(
        self,
        config: GradientAccumulationConfig,
        model: nn.Module,
        optimizer: optim.Optimizer,
        logger: Optional[TrainingLogger] = None
    ):
        
    """__init__ function."""
self.config = config
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.error_handler = ErrorHandler(debug_mode=True)
        
        # Accumulation state
        self.current_step = 0
        self.accumulation_step = 0
        self.accumulated_gradients = {}
        self.accumulation_stats = {
            "total_steps": 0,
            "accumulation_steps": 0,
            "effective_batch_sizes": [],
            "gradient_norms": [],
            "loss_scaling_factors": [],
            "memory_usage": []
        }
        
        # Initialize accumulation
        self._initialize_accumulation()
        
        if self.logger:
            self.logger.log_info(f"Gradient accumulator initialized with {self.config.accumulation_steps} steps")
    
    def _initialize_accumulation(self) -> Any:
        """Initialize gradient accumulation"""
        
        try:
            # Calculate accumulation steps if auto-calculate is enabled
            if self.config.auto_calculate_steps and self.config.effective_batch_size:
                self._calculate_accumulation_steps()
            
            # Validate configuration
            if self.config.validate_accumulation:
                self._validate_config()
            
            # Initialize accumulated gradients
            if self.config.memory_efficient:
                self._initialize_memory_efficient_accumulation()
            else:
                self._initialize_standard_accumulation()
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Accumulation initialization", "initialize_accumulation")
            raise ModelError(f"Failed to initialize gradient accumulation: {str(e)}")
    
    def _calculate_accumulation_steps(self) -> Any:
        """Calculate required accumulation steps based on effective batch size"""
        
        try:
            # Get current batch size from optimizer's param groups
            current_batch_size = self.optimizer.param_groups[0].get('batch_size', 1)
            
            if current_batch_size <= 0:
                raise ValueError("Invalid batch size in optimizer configuration")
            
            # Calculate required accumulation steps
            required_steps = max(1, self.config.effective_batch_size // current_batch_size)
            
            # Update configuration
            self.config.accumulation_steps = required_steps
            
            if self.logger:
                self.logger.log_info(
                    f"Calculated accumulation steps: {required_steps} "
                    f"(effective_batch_size={self.config.effective_batch_size}, "
                    f"current_batch_size={current_batch_size})"
                )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Accumulation steps calculation", "calculate_accumulation_steps")
            raise
    
    def _validate_config(self) -> bool:
        """Validate gradient accumulation configuration"""
        
        try:
            # Validate accumulation steps
            if self.config.accumulation_steps < 1:
                raise ValueError("Accumulation steps must be at least 1")
            
            # Validate effective batch size
            if self.config.effective_batch_size and self.config.effective_batch_size < 1:
                raise ValueError("Effective batch size must be positive")
            
            # Check for potential issues
            if self.config.accumulation_steps > 100:
                if self.logger:
                    self.logger.log_warning(
                        f"Large accumulation steps ({self.config.accumulation_steps}) "
                        "may cause memory issues"
                    )
            
            if self.logger:
                self.logger.log_info("Gradient accumulation configuration validated")
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Configuration validation", "validate_config")
            raise
    
    def _initialize_standard_accumulation(self) -> Any:
        """Initialize standard gradient accumulation"""
        
        try:
            # Initialize accumulated gradients dictionary
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.accumulated_gradients[name] = torch.zeros_like(param.grad) if param.grad is not None else None
            
            if self.logger:
                self.logger.log_info("Standard gradient accumulation initialized")
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Standard accumulation initialization", "initialize_standard_accumulation")
            raise
    
    def _initialize_memory_efficient_accumulation(self) -> Any:
        """Initialize memory-efficient gradient accumulation"""
        
        try:
            # For memory-efficient accumulation, we don't pre-allocate gradients
            # Instead, we accumulate them in-place during training
            self.accumulated_gradients = {}
            
            if self.logger:
                self.logger.log_info("Memory-efficient gradient accumulation initialized")
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Memory-efficient accumulation initialization", "initialize_memory_efficient_accumulation")
            raise
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation"""
        
        try:
            if self.config.scale_loss and self.config.accumulation_steps > 1:
                scaled_loss = loss / self.config.accumulation_steps
                
                if self.logger and self.config.log_accumulation_stats:
                    self.accumulation_stats["loss_scaling_factors"].append(1.0 / self.config.accumulation_steps)
                
                return scaled_loss
            else:
                return loss
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Loss scaling", "scale_loss")
            return loss
    
    def accumulate_gradients(self, loss: torch.Tensor) -> bool:
        """Accumulate gradients from current forward/backward pass"""
        
        try:
            # Scale loss if configured
            scaled_loss = self.scale_loss(loss)
            
            # Backward pass
            scaled_loss.backward()
            
            # Increment accumulation step
            self.accumulation_step += 1
            self.current_step += 1
            
            # Check if we should update parameters
            should_update = self.accumulation_step >= self.config.accumulation_steps
            
            if should_update:
                # Scale gradients if configured
                if self.config.scale_gradients and self.config.accumulation_steps > 1:
                    self._scale_gradients()
                
                # Update parameters
                self.optimizer.step()
                
                # Clear gradients
                if self.config.clear_gradients:
                    self.optimizer.zero_grad()
                
                # Reset accumulation step
                self.accumulation_step = 0
                
                # Record statistics
                self._record_accumulation_stats()
            
            # Record step statistics
            self._record_step_stats(loss.item())
            
            return should_update
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Gradient accumulation", "accumulate_gradients")
            raise ModelError(f"Failed to accumulate gradients: {str(e)}")
    
    def _scale_gradients(self) -> Any:
        """Scale gradients by accumulation steps"""
        
        try:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data /= self.config.accumulation_steps
            
            if self.logger and self.config.log_accumulation_stats:
                self.logger.log_info(f"Gradients scaled by factor: {1.0 / self.config.accumulation_steps}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Gradient scaling", "scale_gradients")
            raise
    
    def _record_accumulation_stats(self) -> Any:
        """Record accumulation statistics"""
        
        try:
            if not self.config.enable_monitoring:
                return
            
            # Calculate effective batch size
            effective_batch_size = self._calculate_effective_batch_size()
            self.accumulation_stats["effective_batch_sizes"].append(effective_batch_size)
            
            # Calculate gradient norms
            if self.config.check_gradient_norms:
                gradient_norm = self._calculate_gradient_norm()
                self.accumulation_stats["gradient_norms"].append(gradient_norm)
            
            # Record memory usage
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
                self.accumulation_stats["memory_usage"].append(memory_usage)
            
            # Update counters
            self.accumulation_stats["total_steps"] += 1
            self.accumulation_stats["accumulation_steps"] += self.config.accumulation_steps
            
            if self.logger and self.config.log_accumulation_stats:
                self.logger.log_info(
                    f"Accumulation completed - "
                    f"Effective batch size: {effective_batch_size}, "
                    f"Gradient norm: {gradient_norm:.6f}" if self.config.check_gradient_norms else ""
                )
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Statistics recording", "record_accumulation_stats")
    
    def _record_step_stats(self, loss_value: float):
        """Record step-level statistics"""
        
        try:
            if not self.config.enable_monitoring:
                return
            
            # Log progress
            if self.logger and self.config.log_accumulation_stats:
                progress = (self.accumulation_step / self.config.accumulation_steps) * 100
                self.logger.log_info(
                    f"Accumulation progress: {progress:.1f}% "
                    f"({self.accumulation_step}/{self.config.accumulation_steps}) - "
                    f"Loss: {loss_value:.6f}"
                )
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Step statistics recording", "record_step_stats")
    
    def _calculate_effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        
        try:
            # Get current batch size from optimizer
            current_batch_size = self.optimizer.param_groups[0].get('batch_size', 1)
            
            # Calculate effective batch size
            effective_batch_size = current_batch_size * self.config.accumulation_steps
            
            return effective_batch_size
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Effective batch size calculation", "calculate_effective_batch_size")
            return 0
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate gradient norm across all parameters"""
        
        try:
            total_norm = 0.0
            param_count = 0
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
            else:
                return 0.0
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Gradient norm calculation", "calculate_gradient_norm")
            return 0.0
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics"""
        
        try:
            stats = self.accumulation_stats.copy()
            
            # Calculate additional statistics
            if stats["effective_batch_sizes"]:
                stats["avg_effective_batch_size"] = np.mean(stats["effective_batch_sizes"])
                stats["max_effective_batch_size"] = np.max(stats["effective_batch_sizes"])
                stats["min_effective_batch_size"] = np.min(stats["effective_batch_sizes"])
            
            if stats["gradient_norms"]:
                stats["avg_gradient_norm"] = np.mean(stats["gradient_norms"])
                stats["max_gradient_norm"] = np.max(stats["gradient_norms"])
                stats["min_gradient_norm"] = np.min(stats["gradient_norms"])
            
            if stats["memory_usage"]:
                stats["avg_memory_usage"] = np.mean(stats["memory_usage"])
                stats["max_memory_usage"] = np.max(stats["memory_usage"])
                stats["min_memory_usage"] = np.min(stats["memory_usage"])
            
            # Add configuration info
            stats["config"] = {
                "accumulation_steps": self.config.accumulation_steps,
                "effective_batch_size": self.config.effective_batch_size,
                "scale_loss": self.config.scale_loss,
                "scale_gradients": self.config.scale_gradients,
                "memory_efficient": self.config.memory_efficient
            }
            
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Statistics retrieval", "get_accumulation_stats")
            return {}
    
    def reset_accumulation(self) -> Any:
        """Reset accumulation state"""
        
        try:
            self.accumulation_step = 0
            self.current_step = 0
            
            # Clear accumulated gradients
            if not self.config.memory_efficient:
                for name in self.accumulated_gradients:
                    if self.accumulated_gradients[name] is not None:
                        self.accumulated_gradients[name].zero_()
            
            if self.logger:
                self.logger.log_info("Gradient accumulation reset")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Accumulation reset", "reset_accumulation")
    
    def cleanup(self) -> Any:
        """Cleanup resources"""
        
        try:
            # Clear accumulated gradients
            self.accumulated_gradients.clear()
            
            # Save statistics if requested
            if self.config.enable_monitoring:
                self._save_accumulation_stats()
            
            if self.logger:
                self.logger.log_info("Gradient accumulator cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Accumulator cleanup", "cleanup")
    
    def _save_accumulation_stats(self) -> Any:
        """Save accumulation statistics to file"""
        
        try:
            stats = self.get_accumulation_stats()
            
            # Create stats directory
            stats_dir = Path("logs/gradient_accumulation")
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            # Save statistics
            stats_file = stats_dir / f"accumulation_stats_{int(time.time())}.json"
            with open(stats_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(stats, f, indent=2)
            
            if self.logger:
                self.logger.log_info(f"Accumulation statistics saved to {stats_file}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Statistics saving", "save_accumulation_stats")


class GradientAccumulationTrainer:
    """Training wrapper with gradient accumulation"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: GradientAccumulationConfig,
        logger: Optional[TrainingLogger] = None
    ):
        
    """__init__ function."""
self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        
        # Initialize gradient accumulator
        self.accumulator = GradientAccumulator(config, model, optimizer, logger)
        
        # Training state
        self.epoch = 0
        self.batch_count = 0
        self.update_count = 0
        
        if self.logger:
            self.logger.log_info("Gradient accumulation trainer initialized")
    
    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with gradient accumulation"""
        
        try:
            inputs, targets = batch_data
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            if hasattr(self.model, 'loss_fn'):
                loss = self.model.loss_fn(outputs, targets)
            else:
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Accumulate gradients
            should_update = self.accumulator.accumulate_gradients(loss)
            
            # Update batch count
            self.batch_count += 1
            
            # Update parameter count if update occurred
            if should_update:
                self.update_count += 1
            
            # Calculate metrics
            metrics = {
                "loss": loss.item(),
                "accumulation_step": self.accumulator.accumulation_step,
                "accumulation_progress": self.accumulator.accumulation_step / self.config.accumulation_steps,
                "should_update": should_update,
                "batch_count": self.batch_count,
                "update_count": self.update_count
            }
            
            # Add accuracy if applicable
            if outputs.dim() > 1:
                accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                metrics["accuracy"] = accuracy
            
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Training step", "train_step")
            raise ModelError(f"Failed to perform training step: {str(e)}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Train for one epoch with gradient accumulation"""
        
        try:
            self.model.train()
            epoch_losses = []
            epoch_accuracies = []
            update_count = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                # Perform training step
                step_metrics = self.train_step(batch_data)
                
                # Record metrics
                epoch_losses.append(step_metrics["loss"])
                if "accuracy" in step_metrics:
                    epoch_accuracies.append(step_metrics["accuracy"])
                
                if step_metrics["should_update"]:
                    update_count += 1
                
                # Log progress
                if self.logger and batch_idx % 10 == 0:
                    self.logger.log_info(
                        f"Epoch {self.epoch}, Batch {batch_idx}/{len(dataloader)} - "
                        f"Loss: {step_metrics['loss']:.6f}, "
                        f"Accumulation: {step_metrics['accumulation_progress']:.1%}"
                    )
            
            # Calculate epoch metrics
            epoch_metrics = {
                "epoch": self.epoch,
                "avg_loss": np.mean(epoch_losses),
                "total_batches": len(dataloader),
                "parameter_updates": update_count,
                "effective_batch_size": self.accumulator._calculate_effective_batch_size()
            }
            
            if epoch_accuracies:
                epoch_metrics["avg_accuracy"] = np.mean(epoch_accuracies)
            
            # Get accumulation statistics
            accumulation_stats = self.accumulator.get_accumulation_stats()
            epoch_metrics["accumulation_stats"] = accumulation_stats
            
            self.epoch += 1
            
            return epoch_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Epoch training", "train_epoch")
            raise ModelError(f"Failed to train epoch: {str(e)}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        
        try:
            stats = {
                "epoch": self.epoch,
                "batch_count": self.batch_count,
                "update_count": self.update_count,
                "accumulation_stats": self.accumulator.get_accumulation_stats()
            }
            
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Training stats retrieval", "get_training_stats")
            return {}
    
    def reset_training(self) -> Any:
        """Reset training state"""
        
        try:
            self.epoch = 0
            self.batch_count = 0
            self.update_count = 0
            self.accumulator.reset_accumulation()
            
            if self.logger:
                self.logger.log_info("Training state reset")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Training reset", "reset_training")
    
    def cleanup(self) -> Any:
        """Cleanup resources"""
        
        try:
            self.accumulator.cleanup()
            
            if self.logger:
                self.logger.log_info("Gradient accumulation trainer cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Trainer cleanup", "cleanup")


# Utility functions
def create_gradient_accumulator(
    model: nn.Module,
    optimizer: optim.Optimizer,
    accumulation_steps: int = 1,
    effective_batch_size: Optional[int] = None,
    logger: Optional[TrainingLogger] = None,
    **kwargs
) -> GradientAccumulator:
    """Create a gradient accumulator with default settings"""
    
    config = GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        effective_batch_size=effective_batch_size,
        **kwargs
    )
    
    return GradientAccumulator(config, model, optimizer, logger)


def create_gradient_accumulation_trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    accumulation_steps: int = 1,
    effective_batch_size: Optional[int] = None,
    logger: Optional[TrainingLogger] = None,
    **kwargs
) -> GradientAccumulationTrainer:
    """Create a gradient accumulation trainer with default settings"""
    
    config = GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        effective_batch_size=effective_batch_size,
        **kwargs
    )
    
    return GradientAccumulationTrainer(model, optimizer, config, logger)


def calculate_optimal_accumulation_steps(
    target_batch_size: int,
    current_batch_size: int,
    max_memory_usage: Optional[float] = None
) -> int:
    """Calculate optimal accumulation steps based on target batch size and memory constraints"""
    
    try:
        # Basic calculation
        required_steps = max(1, target_batch_size // current_batch_size)
        
        # Check memory constraints if provided
        if max_memory_usage and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_ratio = max_memory_usage / current_memory
            
            # Adjust steps based on memory
            memory_adjusted_steps = int(required_steps * memory_ratio)
            required_steps = min(required_steps, memory_adjusted_steps)
        
        return max(1, required_steps)
        
    except Exception as e:
        print(f"Error calculating accumulation steps: {e}")
        return 1


if __name__ == "__main__":
    # Example usage
    
    # Simple model for testing
    class TestModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.linear = nn.Linear(10, 2)
            self.loss_fn = nn.CrossEntropyLoss()
        
        def forward(self, x) -> Any:
            return self.linear(x)
    
    # Create model and optimizer
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create gradient accumulator
    accumulator = create_gradient_accumulator(
        model=model,
        optimizer=optimizer,
        accumulation_steps=4,
        effective_batch_size=128,
        enable_monitoring=True,
        log_accumulation_stats=True
    )
    
    # Simulate training
    for step in range(10):
        # Simulate batch data
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        
        # Forward pass
        outputs = model(inputs)
        loss = model.loss_fn(outputs, targets)
        
        # Accumulate gradients
        should_update = accumulator.accumulate_gradients(loss)
        
        print(f"Step {step}: Loss={loss.item():.6f}, Update={should_update}")
    
    # Get statistics
    stats = accumulator.get_accumulation_stats()
    print(f"Accumulation statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    accumulator.cleanup() 
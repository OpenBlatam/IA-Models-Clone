from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
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
Mixed Precision Training System

Comprehensive mixed precision training implementation using PyTorch's Automatic Mixed Precision (AMP)
with torch.cuda.amp for improved training speed and reduced memory usage.
"""




@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training"""
    
    # AMP settings
    enable_amp: bool = True  # Enable Automatic Mixed Precision
    dtype: torch.dtype = torch.float16  # Precision for mixed precision (float16 or bfloat16)
    enabled: bool = True  # Whether AMP is enabled (can be toggled during training)
    
    # GradScaler settings
    enable_grad_scaler: bool = True  # Enable gradient scaling
    init_scale: float = 2**16  # Initial scale for gradient scaling
    growth_factor: float = 2.0  # Factor to increase scale when no overflow
    backoff_factor: float = 0.5  # Factor to decrease scale when overflow occurs
    growth_interval: int = 2000  # Number of steps between scale increases
    enabled_grad_scaler: bool = True  # Whether grad scaler is enabled
    
    # Performance settings
    cache_enabled: bool = True  # Enable AMP cache for better performance
    autocast_enabled: bool = True  # Enable autocast context
    
    # Memory optimization
    memory_efficient: bool = False  # Use memory-efficient mixed precision
    clear_cache: bool = True  # Clear CUDA cache periodically
    
    # Monitoring
    enable_monitoring: bool = True  # Enable mixed precision monitoring
    log_amp_stats: bool = True  # Log AMP statistics
    track_memory_usage: bool = True  # Track memory usage changes
    
    # Validation
    validate_amp: bool = True  # Validate AMP configuration
    check_compatibility: bool = True  # Check model compatibility with AMP


class MixedPrecisionTrainer:
    """Mixed precision training manager"""
    
    def __init__(
        self,
        config: MixedPrecisionConfig,
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
        
        # Initialize AMP components
        self.scaler = None
        self.amp_stats = {
            "total_steps": 0,
            "amp_steps": 0,
            "scaler_updates": 0,
            "overflow_count": 0,
            "memory_savings": [],
            "training_speedups": [],
            "scale_values": []
        }
        
        # Initialize mixed precision training
        self._initialize_mixed_precision()
        
        if self.logger:
            self.logger.log_info(f"Mixed precision trainer initialized with dtype: {self.config.dtype}")
    
    def _initialize_mixed_precision(self) -> Any:
        """Initialize mixed precision training components"""
        
        try:
            # Validate configuration
            if self.config.validate_amp:
                self._validate_config()
            
            # Check model compatibility
            if self.config.check_compatibility:
                self._check_model_compatibility()
            
            # Initialize GradScaler if enabled
            if self.config.enable_grad_scaler and torch.cuda.is_available():
                self.scaler = GradScaler(
                    init_scale=self.config.init_scale,
                    growth_factor=self.config.growth_factor,
                    backoff_factor=self.config.backoff_factor,
                    growth_interval=self.config.growth_interval,
                    enabled=self.config.enabled_grad_scaler
                )
                
                if self.logger:
                    self.logger.log_info(
                        f"GradScaler initialized with scale: {self.config.init_scale}, "
                        f"growth_factor: {self.config.growth_factor}"
                    )
            
            # Set up autocast context
            if self.config.autocast_enabled:
                self.autocast_context = autocast(
                    enabled=self.config.enabled,
                    dtype=self.config.dtype,
                    cache_enabled=self.config.cache_enabled
                )
            else:
                self.autocast_context = None
            
            if self.logger:
                self.logger.log_info("Mixed precision training components initialized successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Mixed precision initialization", "initialize_mixed_precision")
            raise ModelError(f"Failed to initialize mixed precision training: {str(e)}")
    
    def _validate_config(self) -> bool:
        """Validate mixed precision configuration"""
        
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                if self.logger:
                    self.logger.log_warning("CUDA not available, mixed precision may not work optimally")
            
            # Validate dtype
            if self.config.dtype not in [torch.float16, torch.bfloat16]:
                raise ValueError(f"Unsupported dtype: {self.config.dtype}. Use torch.float16 or torch.bfloat16")
            
            # Validate scaler settings
            if self.config.init_scale <= 0:
                raise ValueError("init_scale must be positive")
            
            if self.config.growth_factor <= 1.0:
                raise ValueError("growth_factor must be greater than 1.0")
            
            if self.config.backoff_factor <= 0 or self.config.backoff_factor >= 1.0:
                raise ValueError("backoff_factor must be between 0 and 1")
            
            if self.logger:
                self.logger.log_info("Mixed precision configuration validated")
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Configuration validation", "validate_config")
            raise
    
    def _check_model_compatibility(self) -> Any:
        """Check if model is compatible with mixed precision training"""
        
        try:
            # Check for operations that might not work well with mixed precision
            incompatible_ops = []
            
            for name, module in self.model.named_modules():
                # Check for specific operations that might cause issues
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    # BatchNorm can work with mixed precision but might need attention
                    if self.logger:
                        self.logger.log_info(f"BatchNorm layer found: {name}")
                
                # Check for custom operations that might not support mixed precision
                if hasattr(module, 'forward') and not hasattr(module, '_amp_forward'):
                    # This is a basic check - more sophisticated checks could be added
                    pass
            
            if incompatible_ops:
                if self.logger:
                    self.logger.log_warning(f"Potentially incompatible operations: {incompatible_ops}")
            
            if self.logger:
                self.logger.log_info("Model compatibility check completed")
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Model compatibility check", "check_model_compatibility")
            raise
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        
        try:
            if not torch.cuda.is_available():
                return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
            
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "allocated": allocated,
                "reserved": reserved,
                "max_allocated": max_allocated
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Failed to get memory usage: {e}")
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    def _clear_cache_if_needed(self) -> Any:
        """Clear CUDA cache if configured"""
        
        try:
            if self.config.clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Failed to clear cache: {e}")
    
    def train_step(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        device: torch.device
    ) -> Dict[str, Any]:
        """Perform a single training step with mixed precision"""
        
        try:
            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Record initial memory usage
            initial_memory = self._get_memory_usage()
            step_start_time = time.time()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with autocast
            if self.autocast_context:
                with self.autocast_context:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Backward pass with gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Unscale gradients for gradient clipping (if needed)
                self.scaler.unscale_(self.optimizer)
                
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Record scaler statistics
                self.amp_stats["scaler_updates"] += 1
                self.amp_stats["scale_values"].append(self.scaler.get_scale())
                
                # Check for overflow
                if self.scaler.get_scale() < self.config.init_scale:
                    self.amp_stats["overflow_count"] += 1
                    if self.logger:
                        self.logger.log_warning("Gradient overflow detected, scale reduced")
            else:
                # Standard backward pass
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            step_time = time.time() - step_start_time
            final_memory = self._get_memory_usage()
            
            # Calculate accuracy
            if outputs.dim() > 1:
                accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
            else:
                accuracy = ((outputs > 0.5) == targets).float().mean().item()
            
            # Calculate memory savings
            memory_savings = {
                "allocated_savings": initial_memory["allocated"] - final_memory["allocated"],
                "reserved_savings": initial_memory["reserved"] - final_memory["reserved"]
            }
            
            # Update statistics
            self.amp_stats["total_steps"] += 1
            if self.config.enabled:
                self.amp_stats["amp_steps"] += 1
            
            self.amp_stats["memory_savings"].append(memory_savings["allocated_savings"])
            self.amp_stats["training_speedups"].append(step_time)
            
            # Clear cache if needed
            self._clear_cache_if_needed()
            
            # Prepare metrics
            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy,
                "step_time": step_time,
                "memory_allocated": final_memory["allocated"],
                "memory_reserved": final_memory["reserved"],
                "memory_savings": memory_savings["allocated_savings"],
                "amp_enabled": self.config.enabled,
                "scaler_scale": self.scaler.get_scale() if self.scaler else None
            }
            
            # Log statistics periodically
            if self.logger and self.config.log_amp_stats and self.amp_stats["total_steps"] % 100 == 0:
                self.logger.log_info(
                    f"AMP Step {self.amp_stats['total_steps']}: "
                    f"Loss={loss.item():.6f}, "
                    f"Time={step_time:.4f}s, "
                    f"Memory={final_memory['allocated']:.2f}GB, "
                    f"Scale={self.scaler.get_scale() if self.scaler else 'N/A'}"
                )
            
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Mixed precision training step", "train_step")
            raise ModelError(f"Failed to perform mixed precision training step: {str(e)}")
    
    def validate_step(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        device: torch.device
    ) -> Dict[str, Any]:
        """Perform a single validation step with mixed precision"""
        
        try:
            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            self.model.eval()
            
            with torch.no_grad():
                # Forward pass with autocast
                if self.autocast_context:
                    with self.autocast_context:
                        outputs = self.model(inputs)
                        loss = loss_fn(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                
                # Calculate accuracy
                if outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = ((outputs > 0.5) == targets).float().mean().item()
            
            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy,
                "amp_enabled": self.config.enabled
            }
            
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Mixed precision validation step", "validate_step")
            raise ModelError(f"Failed to perform mixed precision validation step: {str(e)}")
    
    def get_amp_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics"""
        
        try:
            stats = self.amp_stats.copy()
            
            # Calculate additional statistics
            if stats["memory_savings"]:
                stats["avg_memory_savings"] = np.mean(stats["memory_savings"])
                stats["max_memory_savings"] = np.max(stats["memory_savings"])
                stats["min_memory_savings"] = np.min(stats["memory_savings"])
            
            if stats["training_speedups"]:
                stats["avg_step_time"] = np.mean(stats["training_speedups"])
                stats["min_step_time"] = np.min(stats["training_speedups"])
                stats["max_step_time"] = np.max(stats["training_speedups"])
            
            if stats["scale_values"]:
                stats["avg_scale"] = np.mean(stats["scale_values"])
                stats["min_scale"] = np.min(stats["scale_values"])
                stats["max_scale"] = np.max(stats["scale_values"])
                stats["current_scale"] = stats["scale_values"][-1] if stats["scale_values"] else None
            
            # Calculate efficiency metrics
            if stats["total_steps"] > 0:
                stats["amp_usage_ratio"] = stats["amp_steps"] / stats["total_steps"]
                stats["overflow_ratio"] = stats["overflow_count"] / stats["total_steps"]
            
            # Add configuration info
            stats["config"] = {
                "enable_amp": self.config.enable_amp,
                "dtype": str(self.config.dtype),
                "enabled": self.config.enabled,
                "enable_grad_scaler": self.config.enable_grad_scaler,
                "init_scale": self.config.init_scale,
                "growth_factor": self.config.growth_factor,
                "backoff_factor": self.config.backoff_factor
            }
            
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "AMP statistics retrieval", "get_amp_stats")
            return {}
    
    def enable_amp(self) -> Any:
        """Enable mixed precision training"""
        
        try:
            self.config.enabled = True
            if self.autocast_context:
                self.autocast_context._enabled = True
            
            if self.logger:
                self.logger.log_info("Mixed precision training enabled")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "AMP enable", "enable_amp")
    
    def disable_amp(self) -> Any:
        """Disable mixed precision training"""
        
        try:
            self.config.enabled = False
            if self.autocast_context:
                self.autocast_context._enabled = False
            
            if self.logger:
                self.logger.log_info("Mixed precision training disabled")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "AMP disable", "disable_amp")
    
    def toggle_amp(self) -> Any:
        """Toggle mixed precision training on/off"""
        
        if self.config.enabled:
            self.disable_amp()
        else:
            self.enable_amp()
    
    def reset_stats(self) -> Any:
        """Reset mixed precision statistics"""
        
        try:
            self.amp_stats = {
                "total_steps": 0,
                "amp_steps": 0,
                "scaler_updates": 0,
                "overflow_count": 0,
                "memory_savings": [],
                "training_speedups": [],
                "scale_values": []
            }
            
            if self.logger:
                self.logger.log_info("Mixed precision statistics reset")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "AMP stats reset", "reset_stats")
    
    def save_amp_stats(self, filepath: str):
        """Save mixed precision statistics to file"""
        
        try:
            stats = self.get_amp_stats()
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(stats, f, indent=2)
            
            if self.logger:
                self.logger.log_info(f"AMP statistics saved to {filepath}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "AMP stats saving", "save_amp_stats")
    
    def cleanup(self) -> Any:
        """Cleanup mixed precision training resources"""
        
        try:
            # Save final statistics
            if self.config.enable_monitoring:
                self.save_amp_stats("logs/amp_final_stats.json")
            
            # Clear cache
            self._clear_cache_if_needed()
            
            if self.logger:
                self.logger.log_info("Mixed precision training cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "AMP cleanup", "cleanup")


class MixedPrecisionOptimizer:
    """Mixed precision optimizer wrapper"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: MixedPrecisionConfig,
        logger: Optional[TrainingLogger] = None
    ):
        
    """__init__ function."""
self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        
        # Initialize mixed precision trainer
        self.mp_trainer = MixedPrecisionTrainer(config, model, optimizer, logger)
        
        if self.logger:
            self.logger.log_info("Mixed precision optimizer initialized")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        device: torch.device
    ) -> Dict[str, Any]:
        """Train for one epoch with mixed precision"""
        
        try:
            self.model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                # Perform training step
                step_metrics = self.mp_trainer.train_step(batch_data, loss_fn, device)
                
                # Accumulate metrics
                epoch_loss += step_metrics["loss"]
                epoch_accuracy += step_metrics["accuracy"]
                num_batches += 1
                
                # Log progress
                if self.logger and batch_idx % 10 == 0:
                    self.logger.log_info(
                        f"Batch {batch_idx}/{len(dataloader)}: "
                        f"Loss={step_metrics['loss']:.6f}, "
                        f"Accuracy={step_metrics['accuracy']:.4f}, "
                        f"Time={step_metrics['step_time']:.4f}s"
                    )
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            # Get AMP statistics
            amp_stats = self.mp_trainer.get_amp_stats()
            
            epoch_metrics = {
                "loss": avg_loss,
                "accuracy": avg_accuracy,
                "amp_stats": amp_stats
            }
            
            return epoch_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Mixed precision epoch training", "train_epoch")
            raise ModelError(f"Failed to train epoch with mixed precision: {str(e)}")
    
    def validate_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        device: torch.device
    ) -> Dict[str, Any]:
        """Validate for one epoch with mixed precision"""
        
        try:
            self.model.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_data in dataloader:
                    # Perform validation step
                    step_metrics = self.mp_trainer.validate_step(batch_data, loss_fn, device)
                    
                    # Accumulate metrics
                    total_loss += step_metrics["loss"]
                    total_accuracy += step_metrics["accuracy"]
                    num_batches += 1
            
            # Calculate validation metrics
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            
            validation_metrics = {
                "validation_loss": avg_loss,
                "validation_accuracy": avg_accuracy
            }
            
            return validation_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Mixed precision epoch validation", "validate_epoch")
            raise ModelError(f"Failed to validate epoch with mixed precision: {str(e)}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        
        try:
            return {
                "amp_stats": self.mp_trainer.get_amp_stats(),
                "config": self.config.__dict__
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Training stats retrieval", "get_training_stats")
            return {}
    
    def cleanup(self) -> Any:
        """Cleanup resources"""
        
        try:
            self.mp_trainer.cleanup()
            
            if self.logger:
                self.logger.log_info("Mixed precision optimizer cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Optimizer cleanup", "cleanup")


# Utility functions
def create_mixed_precision_trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    enable_amp: bool = True,
    dtype: torch.dtype = torch.float16,
    enable_grad_scaler: bool = True,
    logger: Optional[TrainingLogger] = None,
    **kwargs
) -> MixedPrecisionTrainer:
    """Create a mixed precision trainer with default settings"""
    
    config = MixedPrecisionConfig(
        enable_amp=enable_amp,
        dtype=dtype,
        enable_grad_scaler=enable_grad_scaler,
        **kwargs
    )
    
    return MixedPrecisionTrainer(config, model, optimizer, logger)


def create_mixed_precision_optimizer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    enable_amp: bool = True,
    dtype: torch.dtype = torch.float16,
    enable_grad_scaler: bool = True,
    logger: Optional[TrainingLogger] = None,
    **kwargs
) -> MixedPrecisionOptimizer:
    """Create a mixed precision optimizer with default settings"""
    
    config = MixedPrecisionConfig(
        enable_amp=enable_amp,
        dtype=dtype,
        enable_grad_scaler=enable_grad_scaler,
        **kwargs
    )
    
    return MixedPrecisionOptimizer(model, optimizer, config, logger)


def check_amp_compatibility(model: nn.Module) -> Dict[str, Any]:
    """Check if a model is compatible with mixed precision training"""
    
    try:
        compatibility_info = {
            "compatible": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check for operations that might not work well with mixed precision
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                compatibility_info["warnings"].append(
                    f"BatchNorm layer '{name}' - ensure running stats are updated correctly"
                )
            
            # Check for custom operations
            if hasattr(module, 'forward') and not hasattr(module, '_amp_forward'):
                # This is a basic check
                pass
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            compatibility_info["warnings"].append("CUDA not available - mixed precision may not work optimally")
        
        # Check for potential issues
        if compatibility_info["warnings"]:
            compatibility_info["recommendations"].append("Review warnings before enabling mixed precision")
        
        return compatibility_info
        
    except Exception as e:
        return {
            "compatible": False,
            "error": str(e),
            "warnings": [],
            "recommendations": ["Check model structure and dependencies"]
        }


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
    
    # Check compatibility
    compatibility = check_amp_compatibility(model)
    print(f"AMP Compatibility: {compatibility}")
    
    # Create mixed precision trainer
    mp_trainer = create_mixed_precision_trainer(
        model=model,
        optimizer=optimizer,
        enable_amp=True,
        dtype=torch.float16,
        enable_grad_scaler=True
    )
    
    # Simulate training
    for step in range(10):
        # Simulate batch data
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        
        # Training step
        metrics = mp_trainer.train_step(
            (inputs, targets),
            lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        print(f"Step {step}: Loss={metrics['loss']:.6f}, Time={metrics['step_time']:.4f}s")
    
    # Get statistics
    stats = mp_trainer.get_amp_stats()
    print(f"AMP Statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    mp_trainer.cleanup() 
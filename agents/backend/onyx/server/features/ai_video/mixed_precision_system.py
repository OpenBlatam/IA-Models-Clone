from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import gc
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Mixed Precision Training System

Comprehensive mixed precision training implementation using torch.cuda.amp
with advanced features for optimal performance and memory efficiency.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    # Basic settings
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    device_type: str = "cuda"
    # Torch performance
    enable_tf32: bool = True
    enable_torch_compile: bool = True
    torch_compile_mode: Optional[str] = None  # None|'default'|'reduce-overhead'|'max-autotune'
    
    # Performance settings
    autocast_enabled: bool = True
    grad_scaler_enabled: bool = True
    cache_enabled: bool = True
    
    # Memory optimization
    memory_efficient: bool = True
    clear_cache: bool = True
    optimize_memory: bool = True
    
    # Advanced features
    dynamic_scaling: bool = True
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    hysteresis: int = 2
    
    # Monitoring
    track_performance: bool = True
    log_scaling: bool = True
    profile_memory: bool = True
    
    # Safety settings
    max_scale: float = 2.0**16
    min_scale: float = 2.0**(-16)
    scale_window: int = 2000
    
    def __post_init__(self) -> Any:
        """Post-initialization setup."""
        if not torch.cuda.is_available():
            self.enabled = False
            self.device_type = "cpu"
            logger.warning("CUDA not available, mixed precision disabled")
        else:
            # Safe CUDA hints
            if self.enable_tf32:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.set_float32_matmul_precision('high')
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                except Exception:
                    pass

class MixedPrecisionManager:
    """Advanced mixed precision manager with comprehensive features."""
    
    def __init__(self, config: MixedPrecisionConfig):
        
    """__init__ function."""
self.config = config
        self.scaler = None
        self.autocast_context = None
        self.performance_metrics = {}
        self.memory_usage = []
        self.scaling_history = []
        
        # Initialize mixed precision components
        self._initialize_mixed_precision()
        
        logger.info(f"Mixed precision manager initialized with config: {config}")
    
    def _initialize_mixed_precision(self) -> Any:
        """Initialize mixed precision components."""
        if not self.config.enabled:
            logger.info("Mixed precision disabled")
            return
        
        # Initialize GradScaler
        if self.config.grad_scaler_enabled:
            self.scaler = amp.GradScaler(
                init_scale=2.0**16,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=self.config.dynamic_scaling
            )
            logger.info("GradScaler initialized")
        
        # Initialize autocast context
        if self.config.autocast_enabled:
            self.autocast_context = amp.autocast(
                device_type=self.config.device_type,
                dtype=self.config.dtype,
                cache_enabled=self.config.cache_enabled
            )
            logger.info("Autocast context initialized")
    
    def _track_memory_usage(self) -> Any:
        """Track memory usage during mixed precision training."""
        if not self.config.profile_memory:
            return
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            self.memory_usage.append({
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'timestamp': time.time()
            })
    
    def _log_scaling_info(self, scale: float, step: int):
        """Log gradient scaling information."""
        if not self.config.log_scaling:
            return
        
        self.scaling_history.append({
            'step': step,
            'scale': scale,
            'timestamp': time.time()
        })
        
        logger.debug(f"Gradient scale at step {step}: {scale}")
    
    def _update_performance_metrics(self, step: int, loss: float, scale: float, 
                                  memory_allocated: float, training_time: float):
        """Update performance metrics."""
        if not self.config.track_performance:
            return
        
        self.performance_metrics[f'step_{step}'] = {
            'loss': loss,
            'scale': scale,
            'memory_allocated_gb': memory_allocated,
            'training_time': training_time,
            'timestamp': time.time()
        }
    
    @contextmanager
    def autocast_context(self) -> Any:
        """Context manager for autocast."""
        if not self.config.enabled or not self.config.autocast_enabled:
            yield
            return
        
        try:
            with self.autocast_context:
                yield
        except Exception as e:
            logger.error(f"Error in autocast context: {e}")
            raise
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if not self.config.enabled or not self.scaler:
            return loss
        
        return self.scaler.scale(loss)
    
    def unscale_optimizer(self, optimizer: torch.optim.Optimizer):
        """Unscale optimizer gradients."""
        if not self.config.enabled or not self.scaler:
            return
        
        self.scaler.unscale_(optimizer)
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with mixed precision."""
        if not self.config.enabled or not self.scaler:
            optimizer.step()
            return
        
        self.scaler.step(optimizer)
    
    def update_scaler(self) -> Any:
        """Update gradient scaler."""
        if not self.config.enabled or not self.scaler:
            return
        
        self.scaler.update()
    
    def get_scale(self) -> float:
        """Get current gradient scale."""
        if not self.config.enabled or not self.scaler:
            return 1.0
        
        return self.scaler.get_scale()
    
    def is_enabled(self) -> bool:
        """Check if mixed precision is enabled."""
        return self.config.enabled and torch.cuda.is_available()
    
    def optimize_memory(self) -> Any:
        """Optimize memory usage."""
        if not self.config.optimize_memory:
            return
        
        if torch.cuda.is_available():
            # Clear cache
            if self.config.clear_cache:
                torch.cuda.empty_cache()
            
            # Force garbage collection
            if self.config.memory_efficient:
                gc.collect()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'enabled': self.config.enabled,
            'current_scale': self.get_scale(),
            'scaling_history': self.scaling_history[-100:] if self.scaling_history else [],
            'memory_usage': self.memory_usage[-100:] if self.memory_usage else [],
            'performance_metrics': self.performance_metrics,
            'config': self.config.__dict__
        }

class MixedPrecisionTrainer:
    """Trainer with integrated mixed precision training."""
    
    def __init__(self, model: nn.Module, config: MixedPrecisionConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.mixed_precision_manager = MixedPrecisionManager(config)
        
        # Optional torch.compile for speedups on PyTorch 2.x
        if getattr(torch, 'compile', None) is not None and config.enable_torch_compile:
            try:
                mode = config.torch_compile_mode
                if mode is None:
                    mode = 'max-autotune' if torch.cuda.is_available() else 'reduce-overhead'
                model = torch.compile(model, mode=mode)
                logger.info(f"Model compiled with torch.compile (mode={mode})")
            except Exception as e:
                logger.warning(f"torch.compile unavailable or failed: {e}")
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        # Performance tracking
        self.training_losses = []
        self.validation_losses = []
        self.scaling_history = []
        
        logger.info("Mixed precision trainer initialized")
    
    def train_step(self, data: torch.Tensor, targets: torch.Tensor, step: int) -> Dict[str, Any]:
        """Single training step with mixed precision."""
        start_time = time.time()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with self.mixed_precision_manager.autocast_context():
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
        
        # Scale loss
        scaled_loss = self.mixed_precision_manager.scale_loss(loss)
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale optimizer
        self.mixed_precision_manager.unscale_optimizer(self.optimizer)
        
        # Step optimizer
        self.mixed_precision_manager.step_optimizer(self.optimizer)
        
        # Update scaler
        self.mixed_precision_manager.update_scaler()
        
        # Calculate accuracy
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == targets).float().mean().item()
        
        # Timing
        training_time = time.time() - start_time
        
        # Track metrics
        current_scale = self.mixed_precision_manager.get_scale()
        memory_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        self.mixed_precision_manager._update_performance_metrics(
            step, loss.item(), current_scale, memory_allocated, training_time
        )
        
        self.mixed_precision_manager._log_scaling_info(current_scale, step)
        self.mixed_precision_manager._track_memory_usage()
        
        # Optimize memory
        self.mixed_precision_manager.optimize_memory()
        
        return {
            'loss': loss.item(),
            'scaled_loss': scaled_loss.item(),
            'accuracy': accuracy,
            'scale': current_scale,
            'memory_allocated_gb': memory_allocated,
            'training_time': training_time
        }
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, Any]:
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move data to device
            data = data.to(self.model.device)
            targets = targets.to(self.model.device)
            
            # Training step
            step_result = self.train_step(data, targets, batch_idx)
            
            total_loss += step_result['loss']
            total_accuracy += step_result['accuracy']
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                           f"Loss: {step_result['loss']:.4f}, "
                           f"Accuracy: {step_result['accuracy']:.4f}, "
                           f"Scale: {step_result['scale']:.2f}, "
                           f"Memory: {step_result['memory_allocated_gb']:.2f} GB")
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'avg_loss': total_loss / num_batches,
            'avg_accuracy': total_accuracy / num_batches,
            'final_scale': self.mixed_precision_manager.get_scale(),
            'performance_stats': self.mixed_precision_manager.get_performance_stats()
        }
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Validate with mixed precision."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            with self.mixed_precision_manager.autocast_context():
                for batch_idx, (data, targets) in enumerate(dataloader):
                    data = data.to(self.model.device)
                    targets = targets.to(self.model.device)
                    
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    
                    predicted = torch.argmax(outputs, dim=1)
                    accuracy = (predicted == targets).float().mean().item()
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'mixed_precision_stats': self.mixed_precision_manager.get_performance_stats(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'scaling_history': self.scaling_history,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'optimizer_state': self.optimizer.state_dict()
        }

class AdaptiveMixedPrecisionManager(MixedPrecisionManager):
    """Adaptive mixed precision manager with dynamic adjustment."""
    
    def __init__(self, config: MixedPrecisionConfig):
        
    """__init__ function."""
super().__init__(config)
        self.adaptation_history = []
        self.performance_threshold = 0.8
        self.adaptation_cooldown = 100
        
    def _adaptive_scaling_decision(self, current_performance: float) -> bool:
        """Make adaptive decision about scaling."""
        if not self.config.dynamic_scaling:
            return True
        
        # Check performance threshold
        if current_performance < self.performance_threshold:
            # Reduce scaling for better stability
            if self.scaler:
                self.scaler._scale = max(self.scaler._scale * 0.9, self.config.min_scale)
            return False
        
        return True
    
    def train_step(self, data: torch.Tensor, targets: torch.Tensor, step: int) -> Dict[str, Any]:
        """Adaptive training step with mixed precision."""
        # Call parent method
        result = super().train_step(data, targets, step)
        
        # Adaptive scaling decision
        if step % self.adaptation_cooldown == 0:
            current_performance = result.get('accuracy', 0.5)
            should_scale = self._adaptive_scaling_decision(current_performance)
            
            self.adaptation_history.append({
                'step': step,
                'performance': current_performance,
                'should_scale': should_scale,
                'current_scale': result['scale'],
                'timestamp': time.time()
            })
        
        return result
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        stats = self.get_performance_stats()
        stats['adaptation_history'] = self.adaptation_history
        stats['performance_threshold'] = self.performance_threshold
        return stats

# Example usage
def example_usage():
    """Example of using mixed precision training system."""
    
    # Create configuration
    config = MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        autocast_enabled=True,
        grad_scaler_enabled=True,
        dynamic_scaling=True,
        memory_efficient=True,
        track_performance=True,
        log_scaling=True
    )
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).cuda()
    
    # Create trainer
    trainer = MixedPrecisionTrainer(model, config)
    
    # Create dummy dataset
    data = torch.randn(100, 784).cuda()
    targets = torch.randint(0, 10, (100,)).cuda()
    dataset = torch.utils.data.TensorDataset(data, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train with mixed precision
    results = trainer.train_epoch(dataloader, epoch=1)
    
    # Print results
    logger.info(f"Training Results: {json.dumps(results, indent=2, default=str)}")
    
    # Get training stats
    stats = trainer.get_training_stats()
    logger.info(f"Training Stats: {json.dumps(stats, indent=2, default=str)}")

match __name__:
    case "__main__":
    example_usage() 
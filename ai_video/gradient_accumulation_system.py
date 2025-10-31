from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
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
Gradient Accumulation System

Comprehensive gradient accumulation implementation for large batch sizes
with advanced features for memory-efficient training and optimal performance.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    # Accumulation settings
    accumulation_steps: int = 4
    effective_batch_size: int = 128
    target_batch_size: int = 512
    
    # Memory optimization
    clear_gradients: bool = True
    memory_efficient: bool = True
    gradient_checkpointing: bool = False
    
    # Performance settings
    sync_gradients: bool = True
    gradient_scaling: bool = True
    automatic_scaling: bool = True
    
    # Monitoring
    track_memory: bool = True
    log_accumulation: bool = True
    profile_accumulation: bool = False
    
    # Advanced features
    dynamic_accumulation: bool = False
    adaptive_accumulation: bool = False
    gradient_clipping: float = 1.0
    warmup_steps: int = 0
    
    def __post_init__(self) -> Any:
        """Post-initialization setup."""
        if self.automatic_scaling:
            self.accumulation_steps = max(1, self.target_batch_size // self.effective_batch_size)
        
        if self.adaptive_accumulation:
            self.dynamic_accumulation = True

class GradientAccumulator:
    """Advanced gradient accumulator for large batch sizes."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
self.config = config
        self.current_step = 0
        self.accumulation_step = 0
        self.total_gradients = 0
        self.memory_usage = []
        self.performance_metrics = {}
        
        # Initialize accumulation state
        self._reset_accumulation_state()
        
        logger.info(f"Gradient accumulator initialized with {config.accumulation_steps} steps")
    
    def _reset_accumulation_state(self) -> Any:
        """Reset accumulation state."""
        self.accumulation_step = 0
        self.current_step = 0
        self.total_gradients = 0
    
    def _track_memory_usage(self) -> Any:
        """Track memory usage during accumulation."""
        if not self.config.track_memory:
            return
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            self.memory_usage.append({
                'step': self.current_step,
                'accumulation_step': self.accumulation_step,
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'timestamp': time.time()
            })
    
    def _log_accumulation_progress(self, loss: float, batch_size: int):
        """Log accumulation progress."""
        if not self.config.log_accumulation:
            return
        
        effective_batch_size = batch_size * (self.accumulation_step + 1)
        progress = (self.accumulation_step + 1) / self.config.accumulation_steps
        
        logger.info(f"Accumulation Step {self.accumulation_step + 1}/{self.config.accumulation_steps} "
                   f"(Progress: {progress:.1%}, Effective Batch: {effective_batch_size}, Loss: {loss:.4f})")
    
    def _should_accumulate(self, model: nn.Module) -> bool:
        """Determine if gradients should be accumulated."""
        # Check if we're in warmup phase
        if self.current_step < self.config.warmup_steps:
            return False
        
        # Check if we've reached accumulation limit
        if self.accumulation_step >= self.config.accumulation_steps - 1:
            return False
        
        # Check memory pressure for adaptive accumulation
        if self.config.adaptive_accumulation:
            return self._check_memory_pressure(model)
        
        return True
    
    def _check_memory_pressure(self, model: nn.Module) -> bool:
        """Check memory pressure for adaptive accumulation."""
        if not torch.cuda.is_available():
            return True
        
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Calculate memory usage percentage
        memory_usage = memory_allocated / total_memory
        
        # If memory usage is high, don't accumulate
        if memory_usage > 0.8:  # 80% threshold
            logger.warning(f"High memory usage detected ({memory_usage:.1%}), skipping accumulation")
            return False
        
        return True
    
    def _scale_loss(self, loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        if not self.config.gradient_scaling:
            return loss
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.config.accumulation_steps
        
        # Additional scaling for effective batch size
        if self.config.automatic_scaling:
            target_scale = self.config.target_batch_size / batch_size
            scaled_loss = scaled_loss * target_scale
        
        return scaled_loss
    
    def _clear_gradients(self, model: nn.Module):
        """Clear gradients efficiently."""
        if not self.config.clear_gradients:
            return
        
        if self.config.memory_efficient:
            # Clear gradients with memory optimization
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
        else:
            # Standard gradient clearing
            model.zero_grad()
    
    def _sync_gradients(self, model: nn.Module):
        """Synchronize gradients across processes if needed."""
        if not self.config.sync_gradients:
            return
        
        # Synchronize gradients for distributed training
        if hasattr(model, 'module'):  # Wrapped model (DataParallel/DistributedDataParallel)
            for param in model.module.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
    
    def _clip_gradients(self, model: nn.Module):
        """Clip gradients for stability."""
        if self.config.gradient_clipping <= 0:
            return
        
        # Calculate total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = self.config.gradient_clipping / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def _update_accumulation_metrics(self, loss: float, batch_size: int, gradient_norm: float):
        """Update accumulation metrics."""
        self.total_gradients += batch_size
        
        # Store performance metrics
        self.performance_metrics[f'step_{self.current_step}'] = {
            'accumulation_step': self.accumulation_step,
            'loss': loss,
            'batch_size': batch_size,
            'effective_batch_size': batch_size * (self.accumulation_step + 1),
            'gradient_norm': gradient_norm,
            'memory_allocated': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            'timestamp': time.time()
        }
    
    @contextmanager
    def accumulation_context(self, model: nn.Module):
        """Context manager for gradient accumulation."""
        try:
            # Track memory usage
            self._track_memory_usage()
            
            # Clear gradients at the start
            self._clear_gradients(model)
            
            yield self
            
        except Exception as e:
            logger.error(f"Error in accumulation context: {e}")
            # Reset accumulation state on error
            self._reset_accumulation_state()
            raise
    
    def accumulate_gradients(self, model: nn.Module, loss: torch.Tensor, 
                           batch_size: int, optimizer: torch.optim.Optimizer,
                           scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, Any]:
        """Accumulate gradients for large batch training."""
        
        # Scale loss for accumulation
        scaled_loss = self._scale_loss(loss, batch_size)
        
        # Backward pass
        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track memory usage
        self._track_memory_usage()
        
        # Log accumulation progress
        self._log_accumulation_progress(loss.item(), batch_size)
        
        # Increment accumulation step
        self.accumulation_step += 1
        
        # Check if we should perform optimizer step
        should_step = not self._should_accumulate(model)
        
        if should_step:
            # Synchronize gradients
            self._sync_gradients(model)
            
            # Clip gradients
            gradient_norm = self._clip_gradients(model)
            
            # Perform optimizer step
            if scaler:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Clear gradients after step
            self._clear_gradients(model)
            
            # Update metrics
            self._update_accumulation_metrics(loss.item(), batch_size, gradient_norm)
            
            # Reset accumulation step
            self.accumulation_step = 0
            self.current_step += 1
            
            # Force garbage collection for memory efficiency
            if self.config.memory_efficient:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return {
            'should_step': should_step,
            'accumulation_step': self.accumulation_step,
            'current_step': self.current_step,
            'effective_batch_size': batch_size * (self.accumulation_step + 1),
            'scaled_loss': scaled_loss.item(),
            'original_loss': loss.item()
        }
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics."""
        return {
            'current_step': self.current_step,
            'accumulation_step': self.accumulation_step,
            'total_gradients': self.total_gradients,
            'effective_batch_size': self.config.effective_batch_size * (self.accumulation_step + 1),
            'target_batch_size': self.config.target_batch_size,
            'memory_usage': self.memory_usage[-10:] if self.memory_usage else [],  # Last 10 entries
            'performance_metrics': self.performance_metrics,
            'config': self.config.__dict__
        }
    
    def reset_stats(self) -> Any:
        """Reset accumulation statistics."""
        self._reset_accumulation_state()
        self.memory_usage.clear()
        self.performance_metrics.clear()

class AdaptiveGradientAccumulator(GradientAccumulator):
    """Adaptive gradient accumulator that adjusts accumulation based on memory and performance."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
super().__init__(config)
        self.adaptation_history = []
        self.performance_threshold = 0.8  # 80% memory usage threshold
        self.adaptation_cooldown = 10  # Steps between adaptations
        
    def _adaptive_accumulation_decision(self, model: nn.Module) -> int:
        """Make adaptive decision about accumulation steps."""
        if not self.config.adaptive_accumulation:
            return self.config.accumulation_steps
        
        # Check memory pressure
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            
            if memory_usage > self.performance_threshold:
                # Reduce accumulation steps
                new_steps = max(1, self.config.accumulation_steps - 1)
                logger.info(f"High memory usage ({memory_usage:.1%}), reducing accumulation steps to {new_steps}")
                return new_steps
            elif memory_usage < 0.5 and self.current_step % self.adaptation_cooldown == 0:
                # Increase accumulation steps
                new_steps = min(self.config.accumulation_steps + 1, self.config.target_batch_size // self.config.effective_batch_size)
                logger.info(f"Low memory usage ({memory_usage:.1%}), increasing accumulation steps to {new_steps}")
                return new_steps
        
        return self.config.accumulation_steps
    
    def accumulate_gradients(self, model: nn.Module, loss: torch.Tensor, 
                           batch_size: int, optimizer: torch.optim.Optimizer,
                           scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, Any]:
        """Accumulate gradients with adaptive adjustment."""
        
        # Adaptive accumulation decision
        if self.config.adaptive_accumulation:
            adaptive_steps = self._adaptive_accumulation_decision(model)
            self.config.accumulation_steps = adaptive_steps
        
        # Call parent method
        result = super().accumulate_gradients(model, loss, batch_size, optimizer, scaler)
        
        # Store adaptation history
        if self.config.adaptive_accumulation:
            self.adaptation_history.append({
                'step': self.current_step,
                'accumulation_steps': self.config.accumulation_steps,
                'memory_usage': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                'timestamp': time.time()
            })
        
        return result
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        stats = self.get_accumulation_stats()
        stats['adaptation_history'] = self.adaptation_history
        stats['performance_threshold'] = self.performance_threshold
        return stats

class GradientAccumulationTrainer:
    """Trainer with integrated gradient accumulation for large batch sizes."""
    
    def __init__(self, model: nn.Module, config: GradientAccumulationConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.accumulator = AdaptiveGradientAccumulator(config) if config.adaptive_accumulation else GradientAccumulator(config)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scaler = torch.cuda.amp.GradScaler() if config.gradient_scaling else None
        
        logger.info(f"Gradient accumulation trainer initialized with {config.accumulation_steps} accumulation steps")
    
    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Single training step with gradient accumulation."""
        batch_size = data.size(0)
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
        
        # Accumulate gradients
        accumulation_result = self.accumulator.accumulate_gradients(
            self.model, loss, batch_size, self.optimizer, self.scaler
        )
        
        # Calculate accuracy
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == targets).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'batch_size': batch_size,
            'effective_batch_size': accumulation_result['effective_batch_size'],
            'should_step': accumulation_result['should_step'],
            'accumulation_step': accumulation_result['accumulation_step']
        }
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move data to device
            data = data.to(self.model.device)
            targets = targets.to(self.model.device)
            
            # Training step with accumulation
            step_result = self.train_step(data, targets)
            
            total_loss += step_result['loss']
            total_accuracy += step_result['accuracy']
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{num_batches}: "
                           f"Loss: {step_result['loss']:.4f}, "
                           f"Accuracy: {step_result['accuracy']:.4f}, "
                           f"Effective Batch: {step_result['effective_batch_size']}")
        
        # Get accumulation stats
        accumulation_stats = self.accumulator.get_accumulation_stats()
        
        return {
            'avg_loss': total_loss / num_batches,
            'avg_accuracy': total_accuracy / num_batches,
            'accumulation_stats': accumulation_stats
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'accumulation_stats': self.accumulator.get_accumulation_stats(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }

# Example usage
def example_usage():
    """Example of using gradient accumulation system."""
    
    # Create configuration
    config = GradientAccumulationConfig(
        accumulation_steps=8,
        effective_batch_size=32,
        target_batch_size=256,
        memory_efficient=True,
        adaptive_accumulation=True,
        gradient_clipping=1.0,
        track_memory=True,
        log_accumulation=True
    )
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).cuda()
    
    # Create trainer
    trainer = GradientAccumulationTrainer(model, config)
    
    # Create dummy dataset
    data = torch.randn(100, 784).cuda()
    targets = torch.randint(0, 10, (100,)).cuda()
    dataset = torch.utils.data.TensorDataset(data, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train with gradient accumulation
    results = trainer.train_epoch(dataloader)
    
    # Print results
    logger.info(f"Training Results: {json.dumps(results, indent=2, default=str)}")
    
    # Get training stats
    stats = trainer.get_training_stats()
    logger.info(f"Training Stats: {json.dumps(stats, indent=2, default=str)}")

match __name__:
    case "__main__":
    example_usage() 